/* Mantainers: 
 * Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz
 */

package isabelle_rl
import java.nio.file.{Path}
import de.unruh.isabelle.control.{Isabelle}
import de.unruh.isabelle.mlvalue.MLValue.{compileValue, compileFunction, compileFunction0}
import de.unruh.isabelle.mlvalue.{MLValue, MLFunction, MLFunction0, MLFunction2, MLFunction3}
import de.unruh.isabelle.mlvalue.{StringConverter, UnitConverter}
import de.unruh.isabelle.pure.{Context, Theory}
import isabelle_rl.Directories


object Writer {
  def apply(logic: String, work_dir: String): Writer = new Writer(logic, work_dir)
}

class Writer (val logic: String, val work_dir: String) {
  override def toString(): String = {
    "Writer(logic=" + logic + ", work_dir=" + work_dir + ")"
  }

  private val setup = Isabelle.Setup(
    isabelleHome = Path.of(Directories.isabelle_repo), 
    logic = this.logic,
    workingDirectory = Path.of(work_dir)
  )

  final val isa_rl_thy_file = Directories.isabelle_rl + "Isabelle_RL.thy"
  implicit val isabelle: Isabelle = new Isabelle(setup)
  private val isabelle_rl_thy : Theory = Theory(Path.of(isa_rl_thy_file))
  private val isa_rl_data = isabelle_rl_thy.importMLStructureNow("Data")

  private val ml_retrieve_from_to = "fn (thy_file, write_dir) => " + s"${isa_rl_data}.retrieve_from_to (\"" + work_dir + "\" ^ thy_file) write_dir"
  final val retrieve_from_to : MLFunction2[String, String, Unit] = compileFunction[String, String, Unit](ml_retrieve_from_to)(isabelle, StringConverter, StringConverter, UnitConverter)

  def data_from_to (logic: String, write_dir: String): Unit = {
    val mlunit = retrieve_from_to(logic, write_dir)(isabelle, StringConverter, StringConverter)
    mlunit.retrieveNow(UnitConverter, isabelle)
  }
}