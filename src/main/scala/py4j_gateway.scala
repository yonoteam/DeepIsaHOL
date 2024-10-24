/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Py4j's gateway server for connecting python with Scala's JVM
*/

package isabelle_rl

import isabelle_rl.Isa_Minion
import isabelle_rl.Writer
import py4j.GatewayServer

object Py4j_Gateway {
  def get_minion(logic: String, work_dir: String): Isa_Minion = new Isa_Minion(logic, work_dir)

  def get_writer(read_dir: String, write_dir: String, logic: String): Writer = new Writer(read_dir, write_dir, logic)

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
