/*  Maintainer: Jonathan Julián Huerta y Munive
    Email: jonjulian23@gmail.com

py4j's gateway server for connecting python with Isabelle/Scala's JVM

NOTE: Need to add py4j's jar file to the classpath
./bin/isabelle scala -cp /Users/jonathan/anaconda3/envs/DeepIsaHOL/share/py4j/py4j0.10.9.7.jar
*/

import py4j.GatewayServer

object Py4j_Gateway {
  private val gatewayServer = new GatewayServer(null)

  def start(args: Array[String]): Unit = {
    gatewayServer.start()
    println("Gateway Server Started")
  }

  def stop(args: Array[String]): Unit = {
    gatewayServer.shutdown()
    println("Gateway Server Stopped")
  }
}