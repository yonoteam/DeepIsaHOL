/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Minion class that does every possible work Isabelle-related 
*/

package isabelle_rl
import scala.jdk.CollectionConverters._
import scala.util.Using
import java.nio.file.{Files, Paths, Path}
import de.unruh.isabelle.control.{Isabelle}
import de.unruh.isabelle.mlvalue.MLValue.{compileValue, compileFunction, compileFunction0}
import de.unruh.isabelle.mlvalue.{MLValue, MLValueWrapper, MLFunction, MLFunction0, MLFunction2, MLFunction3}
import de.unruh.isabelle.mlvalue.{AdHocConverter, StringConverter, UnitConverter}
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

  val setup: Isabelle.Setup = Isabelle.Setup(
    isabelleHome = Path.of(Directories.isabelle_app),
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

  // Isabelle/RL theory (Isabelle_RL.thy) for loading ML files
  val isabelle_rl_thy : Theory = Theory(Path.of(Directories.isabelle_rl))

  // ML_REPL OPERATIONS
  object ML_repl {
    println("just before importing ML structure")
    private final val ml_repl_struct = isabelle_rl_thy.importMLStructureNow("Repl_State")
    println("just after importing ML structure")
    final class Repl_State private (val mlValue: MLValue[Repl_State]) 
      extends MLValueWrapper[Repl_State] {}

    object Repl_State extends MLValueWrapper.Companion[Repl_State] {
      override protected val mlType: String = s"$ml_repl_struct.T" 
      override protected val predefinedException: String = s"$ml_repl_struct.E_Repl_State"
      override protected def instantiate(mlValue: MLValue[Repl_State]): Repl_State = new Repl_State(mlValue)
      
      override protected def newOps(implicit isabelle: Isabelle): Ops = new Ops
      protected class Ops(implicit isabelle: Isabelle) extends super.Ops {
        lazy val repl_test_str = compileValue[String](s"$ml_repl_struct.test_str")
        lazy val repl_init = compileFunction[Theory, Repl_State](s"$ml_repl_struct.init")
        lazy val repl_print = compileFunction[Repl_State, String](s"$ml_repl_struct.print")
        lazy val repl_apply = compileFunction[String, Repl_State, String](s"$ml_repl_struct.repl")
      }
    }
  }

  private object ML_writer {   
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
    val jsons = ML_writer.extract(thy0, thy_text)
    jsons.retrieveNow.split(" ISA_RL_SEP ").toList
  }

  def write_json_proofs (write_dir: Path, thy_file_path: Path): Unit = {
    val thy0 = imports.get_start_theory(thy_file_path)
    val thy_text = imports.get_file_text(thy_file_path)
    ML_writer.write_json_proofs(write_dir.toString(), thy0, thy_text).retrieveNow
  }

  def write_g2tac_proofs (write_dir: Path, thy_file_path: Path): Unit = {
    val thy0 = imports.get_start_theory(thy_file_path)
    val thy_text = imports.get_file_text(thy_file_path)
    ML_writer.write_g2tac_proofs(write_dir.toString(), thy0, thy_text).retrieveNow
  }

  def test_isabelle (ml_code: String = "if true then \"hi Scala\" else \"bye\""): Unit = { 
    val to_print = compileValue[String](ml_code).retrieveNow
    println(to_print)
  }

  def repl_init (thy: Theory): this.ML_repl.Repl_State = ML_repl.Repl_State.Ops.repl_init(thy).retrieveNow

  def repl_apply (txt: String, state: this.ML_repl.Repl_State) = ML_repl.Repl_State.Ops.repl_apply(txt, state).retrieveNow

  def repl_print (state: this.ML_repl.Repl_State) = ML_repl.Repl_State.Ops.repl_print(state).retrieveNow
}