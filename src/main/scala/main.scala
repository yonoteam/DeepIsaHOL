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
import de.unruh.isabelle.pure.{Abs, App, Const, Term, Thm, Transition, Context, Theory}
import isabelle_rl.Directories
import isabelle_rl._

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._


object Main {
  def old_main (args: Array[String]): Unit = {
    val reader = Py4j_Gateway.get_reader(args(0), Directories.test_dir)
    implicit val isabelle = reader.isabelle
    val write_dir = Directories.test_dir + "scala_test/"
    val _ = reader.data_from_to("Test_Thy3.thy", write_dir)
    println("You should see the result now.")
  }

  def main (args: Array[String]): Unit = {
    val logic = "Ordinary_Differential_Equations"
    val reader = Py4j_Gateway.get_reader(logic, Directories.test_dir2)
    implicit val isabelle:de.unruh.isabelle.control.Isabelle = reader.isabelle
    println("Reading file...")
    val jsons = reader.extract("HS_Preliminaries.thy")
    println("The problematic theorem can be printed and it is: \n" + jsons.get(10))
  }
}

