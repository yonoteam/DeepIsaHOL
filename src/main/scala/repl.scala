/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Read-Eval-Print-Loop interface
*/

package isabelle_rl

import scala.collection.mutable.ArrayBuffer

import de.unruh.isabelle.control.{Isabelle}
import de.unruh.isabelle.pure.{Theory}

import isabelle_rl.Repl_State

class REPL(val logic: String = "HOL") {

  // minion
  private val minion: Isa_Minion = new Isa_Minion(logic)
  def get_minion(): Isa_Minion = minion
  implicit val isabelle: Isabelle = minion.isabelle

  // state
  private var state: ArrayBuffer[Repl_State.T] = {
    val local_thys = minion.imports.to_local_list()
    println("Declared local_thys")
    val thy0 = if (local_thys.isEmpty) {
        Theory("Main")
    } else {
        val thy_path = local_thys.head
        minion.imports.get_start_theory(thy_path)
    }
    println("Declared theory")
    ArrayBuffer(minion.repl_init(thy0))
  }

  def read_eval(input: String): Unit = {
    val next_steps = minion.repl_apply(input, state.head)
    state.prependAll(next_steps)
  }

  def print(): String = minion.repl_print(state.head)

  def apply(txt: String): String = {
    read_eval(txt)
    print()
  }

  def reset(): Unit = {
    state.remove(1, state.length - 1)
  }

  def go_back(n: Int): Unit = {
    if (n < state.length) {
      state.dropInPlace(n)
    } else {
      reset()
    }
  }

}