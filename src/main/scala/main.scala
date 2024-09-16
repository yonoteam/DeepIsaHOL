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
import isabelle_rl.{Writer, TheoryLoader}
import isabelle_rl.{Py4j_Gateway_Writer, Py4j_Gateway_Loader}

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._


object Main {
  def old_main (args: Array[String]): Unit = {
    val writer = Py4j_Gateway_Writer.get_writer(args(0))
    implicit val isabelle = writer.isabelle
    val write_dir = Directories.test_dir + "scala_test/"
    val _ = writer.data_from_to("Test_Thy3.thy", write_dir)
    println("You should see the result now.")
  }

  def main (args: Array[String]): Unit = {
    val logic = args(0)
    val writer = Py4j_Gateway_Writer.get_writer(args(0))
    implicit val isabelle:de.unruh.isabelle.control.Isabelle = writer.isabelle

    val loader = new isabelle_rl.TheoryLoader(logic, Directories.isabelle_app, Directories.test_dir)
    // implicit val isabelle:de.unruh.isabelle.control.Isabelle = loader.isabelle
    val source = scala.io.Source.fromFile(Directories.test_dir + "HS_Preliminaries.thy")
    val hs_prelim_text = try source.mkString finally source.close()
    val hs_preliminaries = TheoryLoader.Text(hs_prelim_text, loader.setup.workingDirectory)
    val hs_prelim_thy = loader.getTheory(hs_preliminaries)
    val end_context = hs_prelim_thy.context
    val problem_thm = Thm(end_context, "triangle_norm_vec_le_sum")
    val to_print = problem_thm.prettyRaw(end_context)
    println("The problematic theorem can be printed and it is: " + to_print)
  }
}

