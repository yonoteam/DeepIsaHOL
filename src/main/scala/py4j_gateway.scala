/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Py4j's gateway server for connecting python with Scala's JVM
*/

package isabelle_rl

import isabelle_rl.Writer
import py4j.GatewayServer

object Py4j_Gateway {
  def get_writer(logic: String, read_dir: String): Writer = new Writer(logic, read_dir)

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

/*

import py4j.GatewayServer

object Py4JEntryPoint {
  def getWriter(logic: String, readDir: String): Writer = new Writer(logic, readDir)
  
  def main(args: Array[String]): Unit = {
    val server = new GatewayServer(Py4JEntryPoint)
    server.start()
    println("Gateway Server Started")
  }
}


public class StackEntryPoint {

    private Stack stack;

    public StackEntryPoint() {
      stack = new Stack();
      stack.push("Initial Item");
    }

    public Stack getStack() {
        return stack;
    }

    public static void main(String[] args) {
        GatewayServer gatewayServer = new GatewayServer(new StackEntryPoint());
        gatewayServer.start();
        System.out.println("Gateway Server Started");
    }

}
*/