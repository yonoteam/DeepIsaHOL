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
import isabelle_rl.{Directories, Utils}

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._

/** A class to manage Isabelle-related work
   * @param work_dir working directory where the Isabelle process runs
   * @param logic (a.k.a. built session or heap image) from where to load Isabelle
   * @param imports_dir aiding directory for quickly retrieving Isabelle theories
   */
class Isa_Minion (val work_dir: String, val logic: String, val imports_dir: String) {
  
  // calculates imports_dir based on logic
  def this(logic: String = "HOL") = this(
    work_dir = System.getProperty("user.dir"),
    logic = logic,
    imports_dir = Utils.get_logic_path(logic) match {
      case Some(path) => path.toString
      case None => System.getProperty("user.dir")
    }
  )

  private val session_roots = if (work_dir.contains(Directories.isabelle_afp) && Utils.valid_afp) {
    Seq(Path.of(Directories.isabelle_afp))
  } else Nil

  val setup: Isabelle.Setup = Isabelle.Setup(isabelleHome = Path.of(Directories.isabelle_app),
    sessionRoots = session_roots,
    logic = logic,
    workingDirectory = Path.of(work_dir)
  )
  implicit val isabelle: Isabelle = new Isabelle(setup)
  val imports = Imports(imports_dir)(isabelle)
  imports.start()
  
  override def toString(): String = {
    "Minion(logic=" + logic + ", work_dir=" + work_dir + ", import_dir=" + imports_dir + ")"
  }

  // find the .thy file inside the minion's work directory or its subdirectories
  def get_theory_file_path(file_name: String): Option[Path] = {
    val file_path = Path.of(file_name)
    if (Files.exists(file_path)) {
      Some(file_path)
    } else {
      val proper_file_name = if (file_name.endsWith(".thy")) file_name else {
        file_name + ".thy"
      }
      val path_to_file = imports.local_thy_files.find(_.endsWith(proper_file_name))
      path_to_file
    }
  }

  private object ML_Functions {
    final val isa_rl_thy_file = Directories.isabelle_rl
    val isabelle_rl_thy : Theory = Theory(Path.of(isa_rl_thy_file))

    val ml_actions = isabelle_rl_thy.importMLStructure("Actions")

    val ml_writer = isabelle_rl_thy.importMLStructureNow("Writer")
    final val extract : MLFunction2[Theory, String, String] 
      = compileFunction[Theory, String, String](s"${ml_writer}.extract_jsons")
    final val write_json_proofs : MLFunction3[String, Theory, String, Unit] 
      = compileFunction[String, Theory, String, Unit](s"${ml_writer}.write_json_proofs")
    final val write_g2tac_proofs : MLFunction3[String, Theory, String, Unit] 
      = compileFunction[String, Theory, String, Unit](s"${ml_writer}.write_g2tac_proofs")
  }
  
  def extract (thy_file_path: Path): List[String] = {  
    val thy0 = imports.get_start_theory(thy_file_path)
    val thy_text = imports.get_file_text(thy_file_path)
    val jsons = ML_Functions.extract(thy0, thy_text)
    jsons.retrieveNow.split(" ISA_RL_SEP ").toList
  }

  def write_json_proofs (write_dir: Path, thy_file_path: Path): Unit = {
    val thy0 = imports.get_start_theory(thy_file_path)
    val thy_text = imports.get_file_text(thy_file_path)
    ML_Functions.write_json_proofs(write_dir.toString(), thy0, thy_text).retrieveNow
  }

  def write_g2tac_proofs (write_dir: Path, thy_file_path: Path): Unit = {
    val thy0 = imports.get_start_theory(thy_file_path)
    val thy_text = imports.get_file_text(thy_file_path)
    ML_Functions.write_g2tac_proofs(write_dir.toString(), thy0, thy_text).retrieveNow
  }
}