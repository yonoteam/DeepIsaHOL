/*  Maintainer: Jonathan Julián Huerta y Munive

py4j's gateway server for connecting python with Isabelle/Scala's JVM

STEPS: 
0. add py4j's jar file to the classpath:  isabelle scala -cp /Users/jonathan/anaconda3/envs/DeepIsaHOL/share/py4j/py4j0.10.9.7.jar
1. load this file:                        :load src/py4j_gateway.scala
2. start the gateway:                     val _ = Py4j_Gateway.start(Array("test"));
3. load isabelle:                         import isabelle._
4. import gateway in python:              from py4j.java_gateway import JavaGateway
5. connect to the JVM:                    gateway = JavaGateway()
6. do stuff e.g:                          Isabelle_System.init("","")
7. consult what you can do:               gateway.help(Isabelle_System) or gateway.help(Options)
8. stop in python:                        gateway.shutdown()
9. stop in scala:                         val _ = Py4j_Gateway.stop(Array("test"));
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