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
      isabelleHome = Path.of("..."), 
      logic = "HOL",
      workingDirectory = Paths.get("...")
      )
    implicit val isabelle: Isabelle = new Isabelle(setup)

    val theory_path = "Test_Thy3"
    val some_manager = new TheoryManager()
    println("we have successfully loaded a theory manager")

    // Theory.registerSessionDirectoriesNow("Test_Thy3" -> Path.of("..."))
    // Theory.registerSessionDirectoriesNow("Complex_Main" -> setup.isabelleHome.resolve("src/HOL"))
    
    val some_text = TheoryManager.Text("""theory Test_Thy3 imports Complex_Main begin""", Paths.get("..."))
    println("we got the source")
    some_text.print

    val thy = some_manager.getTheory(some_text).force
    println("we got the theory")

    val get_theory_name = compileFunction[Theory,String]("Context.theory_name {long=true}")
    println("we have defined the theory name")

    println(get_theory_name(thy).retrieveNow)
  }
}

