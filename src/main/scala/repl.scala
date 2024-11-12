/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Read-Eval-Print-Loop interface
*/

package isabelle_rl

class REPL(val logic: String = "HOL") {

  private val minion: Isa_Minion = new Isa_Minion("", logic, "") {

  }

}