package scalakim

import org.apache.mxnet._
import org.apache.mxnet.module.Module
import org.apache.mxnet.NDArray.random_uniform

import scala.collection.mutable.{Map}
import scala.io.{Codec, Source}

import java.nio.charset.CodingErrorAction

import org.slf4j.Logger
import org.slf4j.LoggerFactory

object Inference {
  val usage = """
    Usage: Inference [embedding-file] [word-embedding-dim] [sentence-length] [random-seed (default 1)]
  """
  val logger = LoggerFactory.getLogger(Inference.getClass)
  var embeddingCache:Map[String, Array[Float]] = Map()
  val mod = Module.loadCheckpoint("kim", 100) // modelPrefix="kim", loadModelEpoch=100

  var batchSize    = 1
  var channel      = 1
  var embeddingDim = 300
  var singleSentenceLength = 49
  var embeddingDimShape = Shape(embeddingDim)

  var random = new scala.util.Random(1)

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

  def sentenceToEmbedding(words : Array[String], matrix : NDArray, sentenceIdx : Int): NDArray = {
    var chrIdx = 0
    for (word <- words) {
      if (chrIdx != singleSentenceLength) {
        val wordVec: Option[Array[Float]] = embeddingCache.get(word)
        var vec: NDArray = NDArray.array(wordVec.getOrElse(Array.fill(embeddingDim){random.nextFloat}), shape = embeddingDimShape)
        matrix.at(sentenceIdx).at(0).at(chrIdx).set(vec)
      }
      chrIdx += 1
    }
    return(matrix)
  }

  def batchInference(matrix : NDArray, mod : Module): NDArray = {
    mod.forward(new DataBatch(
      data = IndexedSeq(matrix),
      label = null, index = null, pad = 0))
    return mod.getOutputs()(0)(0)
  }

  def embedding(sentences : Array[Array[String]], dShape: Shape): NDArray = {
    var matrix = NDArray.zeros(dShape)
    
    for (i <- 0 until sentences.length) {
      var words = sentences(i)
      var sentenceMatrix = sentenceToEmbedding(words, matrix, i)
    }

    return matrix
  }

  def main(args : Array[String]) {
    if (args.length < 3) {
      logger.error(usage);
      System.exit(1)
    }

    var embeddingFile = args(0)

    logger.info("Embedding File Path:")
    logger.info(embeddingFile)
    
    embeddingDim = args(1).toInt
    singleSentenceLength = args(2).toInt

    if (args.length == 4) {
      random = new scala.util.Random(args(3).toInt)
    }

    logger.info("Length Limit of a single sentence:")
    logger.info(singleSentenceLength.toString())    
    
    logger.info("Dimension of a single word:")
    logger.info(embeddingDim.toString())

    var dShape = Shape(batchSize, channel, singleSentenceLength, embeddingDim)
    var dataDesc = IndexedSeq(DataDesc("1", dShape)) // "1" is the name of the first operator in the ONNX model -- a convention from ONNX group
    mod.bind(forTraining = false, dataShapes = dataDesc)

    init_embedding(embeddingFile)

    logger.info("Embedding Loaded:")
    logger.info(embeddingCache.size.toString())

    var input_str = "in his first stab at the form , jacquot takes a slightly anarchic approach that works only sporadically ."

    var embedding_generation = embedding(Array(input_str.split(" ")), dShape)
    var output = batchInference(embedding_generation, mod)
    var inference_output_array =  output.at(0).toArray
    var max_value_inference = inference_output_array.max
    val inference_index = inference_output_array.indexOf(max_value_inference)

    logger.info("Input:")
    logger.info(input_str)

    logger.info("Inference Output - Class:")
    logger.info(inference_index.toString())

    logger.info("Inference Output - Vector:")
    logger.info(output.at(0).toArray.map(_.toString()).mkString(" "))

    System.exit(0)
  }
}