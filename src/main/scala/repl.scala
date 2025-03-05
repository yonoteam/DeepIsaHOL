/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Read-Eval-Print-Loop interface
*/

package isabelle_rl

import scala.collection.mutable.ArrayBuffer

import de.unruh.isabelle.control.{Isabelle}
import de.unruh.isabelle.pure.{Theory}
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._

import isabelle_rl._

class REPL(val logic: String = "HOL") {

  // minion
  private val minion: Isa_Minion = new Isa_Minion(logic)
  def get_minion(): Isa_Minion = minion
  implicit val isabelle: Isabelle = minion.isabelle

  // state
  private var state: minion.ML_repl.Repl_State = {
    val local_thys = minion.imports.to_local_list()
    val thy0 = if (local_thys.isEmpty) {
        Theory("Main")
    } else {
        val thy_path = local_thys.head
        minion.imports.get_start_theory(thy_path)
    }
    val start_state = minion.repl_init(thy0)
    println("REPL started!")
    start_state
  }

  def latest_error (): String = {
    minion.repl_latest_error(state)
  }

  def state_size(): Int = {
    minion.repl_size(state)
  }

  def show_curr(): Unit = {
    println(minion.repl_print(state))
  }
  
  def print(): String = {
    minion.repl_print(state)
  }

  def apply(txt: String): String = {
    state = minion.repl_apply(txt, state)
    this.print()
  }

  def shutdown(): Unit = {
    isabelle.destroy()
    sys.exit()
  }
  

  def reset(): Unit = {
    state = minion.repl_reset(state)
  }

  def undo(): Unit = {
    state = minion.repl_undo(state)
  }

  // def go_back(n: Int): Unit = {
  //   if (n < state_size()) {
      
  //   } else {
  //     reset()
  //   }
  // }

}