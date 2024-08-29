/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Main entrypoint
*/

package isabelle_rl

import java.nio.file.{Path, Paths}

import de.unruh.isabelle.control.{Isabelle, OperationCollection}
import de.unruh.isabelle.mlvalue.MLValue.{compileValue, compileFunction, compileFunction0}
import de.unruh.isabelle.mlvalue.{MLValue, MLFunction, MLFunction0, MLFunction2, MLFunction3}
import de.unruh.isabelle.pure.{Abs, App, Const, Term, Transition, Context, Theory}
import isabelle_rl.Directories
import isabelle_rl.Writer
import isabelle_rl.Py4j_Gateway

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._


object Main {
  def main (args: Array[String]): Unit = {
    val writer = Py4j_Gateway.get_writer(args(0))
    implicit val isabelle = writer.isabelle
    val write_dir = Directories.test_dir + "scala_test/"
    val _ = writer.data_from_to("Test_Thy3.thy", write_dir)
    println("You should see the result now.")
  }
}

