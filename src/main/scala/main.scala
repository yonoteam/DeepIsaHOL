package isabelle_rl

import java.nio.file.{Path, Paths}

import de.unruh.isabelle.control.{Isabelle, OperationCollection}
import de.unruh.isabelle.mlvalue.MLValue.{compileValue, compileFunction, compileFunction0}
import de.unruh.isabelle.mlvalue.{MLValue, MLFunction, MLFunction0, MLFunction2, MLFunction3}
import de.unruh.isabelle.pure.{Abs, App, Const, Term, Transition, Context, Theory}
import isabelle_rl.Directories
import isabelle_rl.Writer

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._


object Main {
  def main (args: Array[String]): Unit = {
    val writer = Writer(args(0), Directories.test_dir)
    val write_dir = "/Users/jonathan/Programs/isabelle/learning/ML_Programming/scala_print_test/"
    implicit val isabelle = writer.isabelle
    val _ = writer.data_from_to("Test_Thy3.thy", write_dir).retrieveNow
    println("You should see the result now.")
  }
}

