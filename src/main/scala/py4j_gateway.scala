/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Py4j's gateway server for connecting python with Scala's JVM
*/

package isabelle_rl

import java.nio.file.Path
import py4j.GatewayServer
import scala.sys.ShutdownHookThread
import scala.util.control.NonFatal
import java.util.concurrent.CountDownLatch

import isabelle_rl.Isa_Minion
import isabelle_rl.Writer
import isabelle_rl.REPL
import com.ibm.icu.util.CodePointTrie.Fast

object Py4j_Gateway {
  private val latch = new CountDownLatch(1)
  private val gateway_server = new GatewayServer(Py4j_Gateway)
  private var is_running = false

  // Utilities
  def str_to_path(dir: String): Path = Path.of(dir)
  def path_to_str(path: Path): String = path.toString()

  // Getters
  def get_repl(logic: String="HOL", thy_name:String="Scratch.thy"): REPL 
    = new REPL(logic, thy_name)
  
  def get_minion(work_dir: String, logic: String, imports_dir: String): Isa_Minion 
    = new Isa_Minion(work_dir, logic, imports_dir)

  def get_writer(read_dir: String, write_dir: String, logic: String): Writer 
    = new Writer(read_dir, write_dir, logic)

  // Main
  def start(): Unit = {
    if (is_running) {
      println("Gateway Server is already running")
      return
    }

    try {
      gateway_server.start()
      is_running = true
      println("Gateway Server Started")

      // Keep the process alive and handle shutdown signals
      ShutdownHookThread {
        if (is_running) stop()
        latch.countDown()
      }

      // Prevent the main thread from exiting immediately
      latch.await()
    } catch {
      case NonFatal(e) => 
        println(s"Error starting Gateway Server: ${e.getMessage}")
        e.printStackTrace()
        stop()
    }
  }

  def stop(): Unit = {
    if (!is_running) {
      println("Gateway Server is not running")
      return
    }
    try {
      gateway_server.shutdown()
      is_running = false
      println("Gateway Server Stopped")
    } finally {
      latch.countDown()
    }
  }

  def terminate(): Unit = {
    stop()
    Thread.sleep(500)
    System.exit(0)
  }

  def isServerRunning: Boolean = is_running

  def main(args: Array[String]): Unit = {
    start()
  }
}
