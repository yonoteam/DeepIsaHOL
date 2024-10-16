/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Minion class that does every possible work Isabelle-related 
*/

package isabelle_rl
import scala.jdk.CollectionConverters._
import java.nio.file.{Files, Paths, Path}
import scala.util.Using
import de.unruh.isabelle.control.{Isabelle}
import de.unruh.isabelle.mlvalue.MLValue.{compileValue, compileFunction, compileFunction0}
import de.unruh.isabelle.mlvalue.{MLValue, MLFunction, MLFunction0, MLFunction2, MLFunction3}
import de.unruh.isabelle.mlvalue.{StringConverter, UnitConverter}
import de.unruh.isabelle.pure.{Context, Theory}
import isabelle_rl.{Directories}

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._

object Isa_Minion {
  def apply(logic: String, work_dir: String): Isa_Minion = new Isa_Minion(logic, work_dir)
}

class Isa_Minion (val logic: String, val work_dir: String) {
  private val session_roots = if (work_dir.contains(Directories.isabelle_afp)) {
    Seq(Path.of(Directories.isabelle_afp))
  } else Nil

  val setup: Isabelle.Setup = Isabelle.Setup(isabelleHome = Path.of(Directories.isabelle_app),
    sessionRoots = session_roots,
    logic = logic,
    workingDirectory = Path.of(work_dir)
  )
  implicit val isabelle: Isabelle = new Isabelle(setup)
  val imports = Imports(work_dir)(isabelle)
  imports.start()
  
  override def toString(): String = {
    "Minion(logic=" + logic + ", work_dir=" + work_dir + ")"
  }

  private object ML_Functions {
    final val isa_rl_thy_file = Directories.isabelle_rl + "Isabelle_RL.thy"
    val isabelle_rl_thy : Theory = Theory(Path.of(isa_rl_thy_file))
    val ml_writer = isabelle_rl_thy.importMLStructureNow("Writer")
    final val extract : MLFunction2[Theory, String, String] 
      = compileFunction[Theory, String, String](s"${ml_writer}.extract_jsons")
    final val write_proofs : MLFunction3[String, Theory, String, Unit] 
      = compileFunction[String, Theory, String, Unit](s"${ml_writer}.write_json_proofs")
  }
  
  def extract (thy_file_path: Path): List[String] = {  
    val thy0 = imports.get_start_theory(thy_file_path)
    val thy_text = imports.get_file_text(thy_file_path)
    val jsons = ML_Functions.extract(thy0, thy_text)
    jsons.retrieveNow.split(" ISA_RL_SEP ").toList
  }

  def write_proofs (write_file_path: Path, thy_file_path: Path): Unit = {
    val thy0 = imports.get_start_theory(thy_file_path)
    val thy_text = imports.get_file_text(thy_file_path)
    ML_Functions.write_proofs(write_file_path.toString(), thy0, thy_text).retrieveNow
  }
}