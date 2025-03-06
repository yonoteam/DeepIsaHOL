/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Py4j's gateway server for connecting python with Scala's JVM
*/

package isabelle_rl

import java.nio.file.Path
import isabelle_rl.Isa_Minion
import isabelle_rl.Writer
import isabelle_rl.REPL
import py4j.GatewayServer

object Py4j_Gateway {
  def str_to_path(dir: String): Path = Path.of(dir)

  def path_to_str(path: Path): String = path.toString()

  def get_repl(logic: String="HOL", thy_name:String="Scratch.thy"): REPL = new REPL(logic, thy_name)
  
  def get_minion(work_dir: String, logic: String, imports_dir: String): Isa_Minion = new Isa_Minion(work_dir, logic, imports_dir)

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
