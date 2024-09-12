/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Py4j's gateway server for connecting python with Scala's JVM
*/

package isabelle_rl

import isabelle_rl.Writer
import py4j.GatewayServer

object Py4j_Gateway_Writer {
  def get_writer(logic: String): Writer = new Writer(logic, Directories.test_dir)

  val gateway_server = new GatewayServer(Py4j_Gateway_Writer)

  def start(args: Array[String]): Unit = {
    gateway_server.start()
    println("Gateway Server Started")
  }

  def stop(args: Array[String]): Unit = {
    gateway_server.shutdown()
    println("Gateway Server Stopped")
  }
}

object Py4j_Gateway_Loader {
  def get_writer(logic: String): Writer = new Writer(logic, Directories.test_dir)

  val gateway_server = new GatewayServer(Py4j_Gateway_Loader)

  def start(args: Array[String]): Unit = {
    gateway_server.start()
    println("Gateway Server Started")
  }

  def stop(args: Array[String]): Unit = {
    gateway_server.shutdown()
    println("Gateway Server Stopped")
  }
}
