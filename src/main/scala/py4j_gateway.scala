/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Py4j's gateway server for connecting python with Scala's JVM
*/

package isabelle_rl

import isabelle_rl.Reader
import py4j.GatewayServer

object Py4j_Gateway {
  def get_reader(logic: String, work_dir: String): Reader = Reader(logic, work_dir)

  val gateway_server = new GatewayServer(Py4j_Gateway)

  def start(args: Array[String]): Unit = {
    gateway_server.start()
    println("Gateway Server Started")
  }

  def stop(args: Array[String]): Unit = {
    gateway_server.shutdown()
    println("Gateway Server Stopped")
  }
}
