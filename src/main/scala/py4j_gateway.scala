/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Py4j's gateway server for connecting python with Scala's JVM
*/

package isabelle_rl

import java.nio.charset.StandardCharsets
import java.io.{File, PrintWriter}
import java.nio.file.{Files, Path, Paths, StandardOpenOption}
import java.util.concurrent.CountDownLatch

import scala.util.Random
import scala.util.control.NonFatal

import io.circe._
import io.circe.parser._
import io.circe.syntax._

import py4j.GatewayServer

import isabelle_rl.Isa_Minion
import isabelle_rl.Writer
import isabelle_rl.REPL

class Py4j_Gateway(port: Int = 25333) {
  private val latch = new CountDownLatch(1)
  private val gateway_server = new GatewayServer(this, port)
  private var is_running = false

  // Getters
  def get_repl(logic: String="HOL", thy_name:String="Scratch.thy"): REPL 
    = new REPL(logic, thy_name)
  
  def get_minion(work_dir: String, logic: String, imports_dir: String): Isa_Minion 
    = new Isa_Minion(work_dir, logic, imports_dir)

  def get_writer(read_dir: String, write_dir: String, logic: String): Writer 
    = new Writer(read_dir, write_dir, logic)

  // Start and Stop
  def start(): Unit = {
    if (is_running) {
      println(s"Gateway Server is already running on port $port")
      return
    }

    try {
      gateway_server.start()
      is_running = true
      println(s"Gateway Server Started on port $port")

      // Keep the process alive and handle shutdown signals
      sys.addShutdownHook {
        if (is_running) stop()
        latch.countDown()
      }

      // Prevent the main thread from exiting immediately
      latch.await()
    } catch {
      case NonFatal(e) => 
        println(s"Error starting Gateway Server on port $port: ${e.getMessage}")
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

}

object Py4j_Gateway_Main {
  val ports_file_name = "ports.json"

  // Utilities
  def str_to_path(dir: String): Path = Path.of(dir)
  def path_to_str(path: Path): String = path.toString()

  // ports.json
  def load_ports_json(): Map[Int, Boolean] = {
    val ports_path = Paths.get(ports_file_name)
    if (!Files.exists(ports_path)) return Map()
    
    try {
      val json_content = Files.readString(ports_path, StandardCharsets.UTF_8)
      decode[Map[Int, Boolean]](json_content).getOrElse(Map())
    } catch {
      case _: Exception => Map()
    }
  }

  def write_ports_json(ports_registry: Map[Int, Boolean]): Unit = {
    val json_content = ports_registry.asJson.noSpaces
    Files.write(Paths.get(ports_file_name), json_content.getBytes(StandardCharsets.UTF_8),
      StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
  }

  def register_to_ports_json(port: Int): Unit = {
    val ports_registry = load_ports_json() + (port -> true)
    write_ports_json(ports_registry)

    // ensuring port removal when shutting down
    sys.addShutdownHook {
      deregister_from_ports_json(port)
    }
  }

  def deregister_from_ports_json(port: Int): Unit = synchronized {
    val ports_registry = load_ports_json() - port
    write_ports_json(ports_registry)
    println(s"Removed port $port from registry.")
  }

  // Main
  def main(args: Array[String]): Unit = {
    val port = if (args.length > 0) args(0).toInt else (25333 + Random.nextInt(1000))
    register_to_ports_json(port)
    val gateway = new Py4j_Gateway(port)
    gateway.start()
  }
}
