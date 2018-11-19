package scalakim

import org.apache.mxnet._
import org.apache.mxnet.module.Module
import org.apache.mxnet.NDArray.random_uniform

import scala.collection.mutable.{Map}
import scala.io.{Codec, Source}

import java.nio.charset.CodingErrorAction

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.util.Random.nextFloat

import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
  * bin/run-example SparkPageRank data/mllib/pagerank_data.txt 10
  */

object SparkInference {
  def main(args: Array[String]) {
    // Initial Logger & Read Model
    val logger = LoggerFactory.getLogger(SparkInference.getClass)
    var embeddingCache:Map[String, Array[Float]] = Map()

    val batchSize    = 1
    val channel      = 1
    var embeddingDim = 300
    var singleSentenceLength = 49
    var embeddingDimShape = Shape(embeddingDim)

    // Read arguments
    if (args.length != 5) {
      System.err.println("Usage: SparkInference [embedding-file] [word-embedding-dim] [sentence-length] [input-file] [output-file]")
      System.exit(1)
    }

    val embeddingFile = args(0)

    logger.info("Embedding File Path:")
    logger.info(embeddingFile)

    embeddingDim = args(1).toInt
    singleSentenceLength = args(2).toInt
    val inputPath = args(3)
    val outputPath = args(4)

    logger.info("Length Limit of a single sentence:")
    logger.info(singleSentenceLength.toString())

    logger.info("Dimension of a single word:")
    logger.info(embeddingDim.toString())

    logger.info("Input Path:")
    logger.info(inputPath)

    // Cast embedding map
    val conf = new SparkConf().setAppName("SparkInference")
    val sc = new SparkContext(conf)

    def init_embedding(embeddingFile: String): Unit = {
      val decoder = Codec.UTF8.decoder.onMalformedInput(CodingErrorAction.IGNORE)
      for (embeddingStr <- Source.fromFile(embeddingFile)(decoder).getLines) {
        val vec = new Array[Float](embeddingDim)
        var tokens : Array[String] = embeddingStr.split(" ")
        var i = 0
        var word = ""
        for (token <- tokens) {
          if (i == 0) {
            word = token
          } else {
            vec(i-1) = token.toFloat
          }
          i += 1
        }
        embeddingCache(word) = vec
      }
    }

    init_embedding(embeddingFile)

    logger.info("Embedding Loaded:")
    logger.info(embeddingCache.size.toString())

    val bcEmbeddingMap = sc.broadcast(embeddingCache)

    // Cast Model
    var dShape = Shape(batchSize, channel, singleSentenceLength, embeddingDim)
    var brDShape = sc.broadcast(dShape)

    // batch inference
    val outputFDD = sc.textFile(inputPath)
      .mapPartitions(lines => {
        // the model file must exist on each node of the cluster
        val mod = Module.loadCheckpoint("kim", 100) // modelPrefix="kim", loadModelEpoch=100

        // "1" is the name of the first operator in the ONNX model -- a convention from ONNX group
        var dataDesc = IndexedSeq(DataDesc("1", brDShape.value))
        mod.bind(forTraining = false, dataShapes = dataDesc)

        lines.map(input_str => {
          // Build Input Matrix for each input_str
          val matrix = NDArray.zeros(brDShape.value)
          val singleSentenceLength = brDShape.value(2)
          val embeddingLength = brDShape.value(3)
          val words = input_str.split(" ")
          var wordIdx = 0

          // fill in the word vector for each word in input_str
          for (word <- words) {
            if (wordIdx != singleSentenceLength) {
              val wordVec: Option[Array[Float]] = bcEmbeddingMap.value.get(word)
              var vec: NDArray = NDArray.array(wordVec.getOrElse(Array.fill(embeddingLength){nextFloat}), shape = embeddingDimShape)

              matrix.at(0).at(0).at(wordIdx).set(vec) // matrix.at(batch).at(channel)
            }
            wordIdx += 1
          }

          mod.forward(new DataBatch(
            data = IndexedSeq(matrix),
            label = null, index = null, pad = 0))
          var output = mod.getOutputs()(0)(0)

          // get final class index
          var inference_output_array =  output.at(0).toArray
          var max_value_inference = inference_output_array.max
          val inference_index = inference_output_array.indexOf(max_value_inference)
          (input_str, inference_output_array.map(_.toString()).mkString(" "), inference_index)
        })
      })

    outputFDD.saveAsTextFile(outputPath)
  }
}