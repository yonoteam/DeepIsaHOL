/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Read-Eval-Print-Loop interface
*/

package isabelle_rl

import java.nio.file.{Files, Path}

import scala.collection.mutable.ArrayBuffer

import de.unruh.isabelle.control.{Isabelle}
import de.unruh.isabelle.pure.{Theory}
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._

import isabelle_rl._

class REPL(val logic: String = "HOL", thy_name: String = "Scratch.thy") {
  // INITIALIZATION

  // minion
  private val minion: Isa_Minion = new Isa_Minion(logic)
  def get_minion(): Isa_Minion = minion
  implicit val isabelle: Isabelle = minion.isabelle

  // current theory path
  private var current_thy_path: Option[Path] = minion.get_theory_file_path(thy_name)

  // state
  private var state: minion.ML_repl.Repl_State = {
    go_to_end_of(thy_name)
    state
  }
  println("REPL started!")

  def go_to_end_of(thy_name: String): Unit = {
    current_thy_path = minion.get_theory_file_path(thy_name)
    val thy0 = current_thy_path match {
      case Some(path) => minion.imports.get_end_theory(path)
      case None => 
        println(s"Warning: the minion did not find $thy_name. Defaulting to theory 'Main'.")
        Theory("Main")
    }
    state = minion.repl_init(thy0)
  }


  // OPERATIONS

  def apply(txt: String): String = {
    state = minion.repl_apply(txt, state)
    this.state_string()
  }

  def reset(): Unit = {
    state = minion.repl_reset(state)
  }

  def undo(): Unit = {
    state = minion.repl_undo(state)
  }

  def shutdown_isabelle(): Unit = {
    isabelle.destroy()
    print("Isabelle process destroyed.")
  }

  
  // INFORMATION RETRIEVAL
  def isabelle_exists(): Boolean = {
    !(minion.isabelle.isDestroyed)
  }

  def state_string(): String = {
    minion.repl_print(state)
  }

  def state_size(): Int = {
    minion.repl_size(state)
  }

  def latest_error(): String = {
    minion.repl_get_error(state)
  }

  def last_usr_state(): String = {
    minion.repl_last_usr_st(state)
  }

  def show_curr(): Unit = {
    println(this.state_string())
  }
  
  def is_at_proof(): Boolean = {
    minion.repl_is_at_proof(state)
  }

  def without_subgoals(): Boolean = {
    minion.repl_without_subgoals(state)
  }

  def proof_so_far(): String = {
    minion.repl_proof_so_far(state)
  }

  def last_proof(): String = {
    minion.repl_last_proof_of(state)
  }

  // def go_back(n: Int): Unit = {
  //   if (n < state_size()) {
      
  //   } else {
  //     reset()
  //   }
  // }

}