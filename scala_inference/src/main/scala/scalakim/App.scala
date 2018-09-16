package scalakim

import org.apache.mxnet._
import org.apache.mxnet.module.Module
import org.apache.mxnet.NDArray.random_uniform

import scala.collection.mutable.{Map}
import scala.io.{Codec, Source}

import java.nio.charset.CodingErrorAction

object App {
  var embeddingCache:Map[String, NDArray] = Map()
  val mod = Module.loadCheckpoint("kim", 100) // modelPrefix="kim", loadModelEpoch=100
  var dShape = Shape(1, 1, 49, 300)
  var dataDesc = IndexedSeq(DataDesc("1", dShape))
  var batchSize = dShape(0)
  var singleSentenceLength = dShape(2)
  mod.bind(forTraining = false, dataShapes = dataDesc)

  def init_embedding(): Unit = {
    val decoder = Codec.UTF8.decoder.onMalformedInput(CodingErrorAction.IGNORE)
    val filename = "word2vec.txt" // word embeddings
    for (embeddingStr <- Source.fromFile(filename)(decoder).getLines) {
      var vec: NDArray = NDArray.zeros(300)
      var tokens : Array[String] = embeddingStr.split(" ")
      var i = 0
      var word = ""
      for (token <- tokens) {
        if (i == 0) {
          word = token
        } else {
          vec.at(i - 1).set(token.toFloat)
          i += 1
        }
      }
      embeddingCache(word) = vec
    }
  }

  def sentenceToEmbedding(words : Array[String], length : Int, matrix : NDArray, sentenceIdx : Int): NDArray = {
    var chrIdx = 0
    for (word <- words) {
      if (chrIdx != length) {
        val wordVec: Option[NDArray] = embeddingCache.get(word)
        var vec: NDArray = wordVec.getOrElse(random_uniform(-1, 1, 300))
        matrix.at(sentenceIdx).at(0).at(chrIdx).set(vec)
      }
      chrIdx += 1
    }
    return(matrix)
  }

  def batchInference(matrix : NDArray, sentencesSize : Int, singleSentenceLength : Int, mod : Module, dShape : Shape): NDArray = {
    mod.forward(new DataBatch(
      data = IndexedSeq(matrix),
      label = null, index = null, pad = 0))
    return mod.getOutputs()(0)(0)
  }

  def embedding(sentences : Array[Array[String]], length : Int, singleSentenceLength : Int): NDArray = {
    var dShape = Shape(length, 1, singleSentenceLength, 300)
    var matrix = NDArray.zeros(dShape)

    for (i <- 0 until sentences.length) {
      var words = sentences(i)
      var sentenceMatrix = sentenceToEmbedding(words, singleSentenceLength, matrix, i)
    }

    return matrix
  }

  def main(args : Array[String]) {
    init_embedding()
    var input_str = "in his first stab at the form , jacquot takes a slightly anarchic approach that works only sporadically ."
    var embedding_generation = embedding(Array(input_str.split(" ")), 1, 49)
    var output = batchInference(embedding_generation, batchSize, singleSentenceLength, mod, dShape)
    var inference_output_array =  output.at(0).toArray
    var max_value_inference = inference_output_array.max
    val inference_index = inference_output_array.indexOf(max_value_inference)

    println("Inference Output - Class:")
    println(inference_index)

    println("Inference Output - Vector:")
    println(output.at(0).toArray.map(_.toString()).mkString(" "))
  }
}