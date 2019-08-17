
import java.util.Properties

import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}
import play.api.libs.json.Json

import scala.io.Source

object Stream {
  def main(args: Array[String]): Unit = {

    val url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MSFT&interval=1min&apikey=demo"
    val data = Source.fromURL(url)("UTF-8").mkString

    val parsedData = Json.parse(data)

    val open = parsedData \\ "1. open"
    val high = parsedData \\ "2. high"
    val low = parsedData \\ "3. low"
    val close = parsedData \\ "4. close"
    val volume = parsedData \\ "5. volume"

    val props = new Properties()
    props.put("bootstrap.servers", "localhost:9092")
    props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    props.put("value.serializer.encoding", "ISO-8859-1")

    val topic = "testTopic2"
    val producer = new KafkaProducer[String, String](props)

    var i = 0
    try {
      while
      (open(i) != null) {
        val str : String = open(i).as[String].replace("\"","").toFloat + " " +
          high(i).as[String].replace("\"","").toFloat + " " +
          low(i).as[String].replace("\"","").toFloat +" " +
          close(i).as[String].replace("\"","").toFloat + " " +
          volume(i).as[String].replace("\"","").toFloat

        val record = new ProducerRecord(topic, i.toString, str)
        producer.send(record)
        i += 1
        Thread.sleep(1000)
      }
    } catch {
      case aexception: ArrayIndexOutOfBoundsException => println("all elements sent")
      case exception: Exception => println("exception =>  " + exception)
    } finally {
      producer.close()
    }

//    var i = 0
//    while(open(i) != null) {
//      if((open(i).as[String])== "136.3850") {
//        println("string matched")
//      }
//      i += 1
//    }
//    producer.close()

  }
}
