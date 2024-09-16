/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Writes proof data from input theory file
*/

package isabelle_rl
import java.nio.file.{Path}
import de.unruh.isabelle.control.{Isabelle}
import de.unruh.isabelle.mlvalue.MLValue.{compileValue, compileFunction, compileFunction0}
import de.unruh.isabelle.mlvalue.{MLValue, MLFunction, MLFunction0, MLFunction2, MLFunction3}
import de.unruh.isabelle.mlvalue.{StringConverter, UnitConverter}
import de.unruh.isabelle.pure.{Context, Theory}
import isabelle_rl.{Directories, TheoryLoader}

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._

object Writer {
  def apply(logic: String, work_dir: String): Writer = new Writer(logic, work_dir)
}

class Writer (val logic: String, val work_dir: String) {
  override def toString(): String = {
    "Writer(logic=" + logic + ", work_dir=" + work_dir + ")"
  }

  private val loader = new TheoryLoader(logic, Directories.isabelle_app, work_dir)
  implicit val isabelle: Isabelle = loader.isabelle

  private object ML_Functions {
    final val isa_rl_thy_file = Directories.isabelle_rl + "Isabelle_RL.thy"
    val isabelle_rl_thy : Theory = Theory(Path.of(isa_rl_thy_file))
    val isa_rl_data = isabelle_rl_thy.importMLStructureNow("Data")

    final val ml_retrieve_from_to = "fn (thy_file, write_dir) => " + s"${isa_rl_data}.retrieve_from_to (\"" + work_dir + "\" ^ thy_file) write_dir"
    final val retrieve_from_to : MLFunction2[String, String, Unit] = compileFunction[String, String, Unit](ml_retrieve_from_to)
    final val ml_extract = s"${isa_rl_data}.extract"
    final val extract : MLFunction2[Theory, String, String] = compileFunction[Theory, String, String](ml_extract)
  }

  
  def extract (thy_file: String): List[String] = {
    val thy_source = scala.io.Source.fromFile(thy_file)
    val thy_text = try thy_source.mkString finally thy_source.close()
    val thy0 = loader.beginTheory(TheoryLoader.Text(thy_text, loader.setup.workingDirectory))
    val jsons = ML_Functions.extract(thy0, thy_file)
    jsons.retrieveNow.split(" ISA_RL_SEP ").toList
  }

  // brittle on the Isabelle/ML side, better to use extract above or data-retrieve below
  def data_from_to (thy_file: String, write_dir: String): Unit = {
    val mlunit = ML_Functions.retrieve_from_to(thy_file, write_dir)(isabelle, StringConverter, StringConverter)
    mlunit.retrieveNow(UnitConverter, isabelle)
  }

  /* def write_data_to_dir(write_dir: String): Unit {
    println("hello")
  } */
}