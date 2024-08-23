package isabelle_rl

import java.nio.file.{Path, Paths}

import de.unruh.isabelle.control.{Isabelle, OperationCollection}
import de.unruh.isabelle.mlvalue.MLValue.{compileFunction, compileFunction0}
import de.unruh.isabelle.mlvalue.{MLValue, MLFunction, MLFunction0, MLFunction2, MLFunction3}
import de.unruh.isabelle.pure.{Abs, App, Const, Term, Transition, Context, Theory}
import isabelle_rl.TheoryManager._

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._


object Main {
  def main (args: Array[String]): Unit = {
    val setup = Isabelle.Setup(
      isabelleHome = Path.of("/Users/jonathan/Programs/deepIsaHOL/lib/isabelle"), 
      logic = "HOL",
      workingDirectory = Paths.get("/Users/jonathan/Programs/isabelle/learning/ML_Programming/")
      )
    implicit val isabelle: Isabelle = new Isabelle(setup)
    val theory_path = "/Users/jonathan/Programs/isabelle/learning/ML_Programming/Test_Thy4"
    val some_manager = new TheoryManager()
    println("we have successfully looaded a theory manager")
    val thy = some_manager.getTheory(some_manager.getTheorySource(theory_path))
    println("we got the theory")
    val get_theory_name = compileFunction[Theory,String]("Context.theory_name {long=true}")
    println("we have defined the theory name")
    println(get_theory_name(thy).retrieveNow)
  }
}

