package isabelle_rl

import java.nio.file.{Path, Paths}

import de.unruh.isabelle.control.{Isabelle, OperationCollection}
import de.unruh.isabelle.mlvalue.MLValue.{compileValue, compileFunction, compileFunction0}
import de.unruh.isabelle.mlvalue.{MLValue, MLFunction, MLFunction0, MLFunction2, MLFunction3}
import de.unruh.isabelle.pure.{Abs, App, Const, Term, Transition, Context, Theory}
import isabelle_rl.TheoryManager._
import isabelle_rl.Directories

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._


object Main {
  def main (args: Array[String]): Unit = {
    val setup = Isabelle.Setup(
      isabelleHome = Path.of(Directories.isabelle_repo), 
      logic = "HOL",
      workingDirectory = Paths.get(Directories.test_dir)
      )
    implicit val isabelle: Isabelle = new Isabelle(setup)

    val isabelle_rl_thy : Theory = Theory(Path.of(Directories.isabelle_rl + "Isabelle_RL.thy"))
    val rl_ops : String = isabelle_rl_thy.importMLStructureNow("Ops")
    // val Actions : String = isabelle_rl_thy.importMLStructureNow("Actions")
    val read_file : MLFunction[String, String] = compileFunction(s"${rl_ops}.read_file")
    
    println(read_file(Directories.test_dir + "Test_Thy3.thy").retrieveNow)
    /*
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
    */
  }
}

