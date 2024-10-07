/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Main entrypoint
*/

package isabelle_rl

import isabelle_rl.Directories
import isabelle_rl._


object Main {
  def main (args: Array[String]): Unit = {
    val logic = "Framed_ODEs"
    val writer = Py4j_Gateway.get_writer(Directories.test_read_dir, Directories.test_write_dir, logic)
    println("Initialised writer")
    val minion = writer.get_minion()
    implicit val isabelle:de.unruh.isabelle.control.Isabelle = minion.isabelle
    writer.write_all()
  }
}

