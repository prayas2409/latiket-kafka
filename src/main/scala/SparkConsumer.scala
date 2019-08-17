import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010._
import org.apache.spark.streaming.{Seconds, StreamingContext}

object SparkConsumer {
  def main(args: Array[String]): Unit = {
    val ss = SparkSession.builder()
      .appName("consumer")
      .master("local[*]")
      .getOrCreate()

    val scc = new StreamingContext(ss.sparkContext, Seconds(2))


      val kafkaParams = Map[String, Object](
        "bootstrap.servers" -> "localhost:9092",
        "key.deserializer"->"org.apache.kafka.common.serialization.StringDeserializer", // serialize ments to convert to byte stream
        "value.deserializer"->"org.apache.kafka.common.serialization.StringDeserializer",
        "group.id"-> "group5" // clients can take
//        "value.serializer.encoding" -> "UTF-8"
      )

    val topics = Array("testTopic2")

    val stream = KafkaUtils.createDirectStream[String, String](
      scc,
      PreferConsistent,
      Subscribe[String, String](topics, kafkaParams)
    )
    val pyPath = "/home/admin1/IdeaProjects/latiket_kafka/src/main/python/Predictor.py"
    val mappedData = stream.map(x => (x.value().toString))
    mappedData.foreachRDD(
      x =>
//        ss.sqlContext.read.json(x).rdd.collect().foreach(y=>
//          ss.sparkContext.makeRDD(List(y.toString().replaceAll(","," ").replace("[","").replace("]","")))
//            x.collect().foreach(y => ss.sparkContext.makeRDD(List(y)).pipe(pyPath).collect().foreach(println))
          x.collect().foreach(y => ss.sparkContext.makeRDD(List(y)).pipe(pyPath).collect().foreach(println))

    //              x.pipe(pyPath).collect().foreach(println)

    //        )
    )

//    val mappedData = stream.map(x => (x.value().toString))
//    mappedData.foreachRDD(x =>
////      ss.sparkContext.makeRDD((x.collect().toString),1).collect().foreach(println)
//      x.collect().foreach(
//        z =>
//          ss.sparkContext.makeRDD(new String(z.getBytes(),"UTF8")).pipe(pyPath).collect().foreach(println)
////        println("ii:",z.getClass)
//      )
//
//    )

//    stream.map(
//      rdd => new String((rdd.toString).getBytes(StandardCharsets.UTF_8),StandardCharsets.UTF_8).replace("?",""))
//      .foreachRDD(
//        x =>
////          sparkContext.makeRDD(x.pipe(pypath).collect(),1).saveAsTextFile(s3Path)
//                x.pipe(pyPath).collect().foreach(println)
//      )


//    val consumedData = stream.map(record=>(record.value().toString))
//    consumedData.foreachRDD(x =>
////      ss.sparkContext.makeRDD(x.collect().toString.replace("\"",""))
////        .pipe("/home/admin1/IntelliJProject/kafkaStreaming/src/main/scala/stockPredictor.py")
////      .collect().foreach(println)
//      x.collect().foreach(println)
//    )

    scc.start()
    scc.awaitTermination()

//    val kafkaParams = Map("metadata.broker.list" -> "localhost:9092")
//    val topic = Set("testTopic")
//    val stream = KafkaUtils.createDirectStream(stc, kafkaParams, topic)

//    val kafkaParams = Map("metadata.broker.list" -> "localhost:9092,anotherhost:9092")
//    val topics = Set("testTopic")
//    val stream = KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](
//      stc, kafkaParams, topics)
//    val lines = KafkaUtils.createDirectStream(stc, kafkaParams, topics)

//    val lines = KafkaUtils.createDirectStream(stc, "localhost:2181", "consumer-group", Map("testTopic" -> 5))
  }
}
