package isabelle_rl

import java.nio.file.{Path, Paths}

import de.unruh.isabelle.mlvalue.MLValue.{compileFunction, compileFunction0}
import de.unruh.isabelle.mlvalue.{MLValue, MLFunction, MLFunction0, MLFunction2, MLFunction3}
import de.unruh.isabelle.pure.{Abs, App, Const, Term, Transition, Context}
import de.unruh.isabelle.control.Isabelle

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._


object Main {

  
  def main (args: Array[String]): Unit = {
    val setup = Isabelle.Setup(isabelleHome = Path.of("/Applications/Isabelle2023.app"), logic = "HOL")
    implicit val isabelle: Isabelle = new Isabelle(setup)
    println("Testing SBT")
  }
}

