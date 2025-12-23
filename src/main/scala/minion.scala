/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Minion class that does every possible work Isabelle-related 
*/

package isabelle_rl
import scala.sys.process._
import scala.jdk.CollectionConverters._
import java.io.File
import java.nio.file.{Files, Path}
import de.unruh.isabelle.control.{Isabelle}
import de.unruh.isabelle.mlvalue.MLValue.{compileValue, compileFunction}
import de.unruh.isabelle.mlvalue.{MLValue, MLValueWrapper, MLFunction, MLFunction2, MLFunction3}
import de.unruh.isabelle.pure.{Theory}
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._


// MINION CLASS
/** A class to manage Isabelle-related work
   * @param work_dir working directory where the Isabelle process runs
   * @param logic (a.k.a. built session or heap image) from where to load Isabelle
   * @param imports_dir aiding directory for quickly retrieving Isabelle theories
   */
class Isa_Minion (val work_dir: String, val logic: String, val imports_dir: String) {
  // CONSTRUCTORS
  def this(logic: String = "HOL") = this(
    work_dir = System.getProperty("user.dir"),
    logic = logic,
    imports_dir = Utils.get_logic_path(logic) match {
      case Some(path) => path.toString
      case None => System.getProperty("user.dir")
    }
  )


  // MINION CONFIGURATION
  private val session_roots = if (work_dir.contains(Directories.isabelle_afp) && Utils.valid_afp) {
    Seq(Path.of(Directories.isabelle_afp))
  } else if (Utils.is_isa_root_dir(Path.of(work_dir))) {
    Seq(Path.of(work_dir))
  } else {
    Nil
  }

  val setup: Isabelle.Setup = Isabelle.Setup(
    isabelleHome = Path.of(Directories.isabelle_app),
    sessionRoots = session_roots,
    logic = logic,
    workingDirectory = Path.of(work_dir)
  )
  implicit val isabelle: Isabelle = new Isabelle(setup)
  val imports = Imports(imports_dir)(isabelle)
  imports.start()

  // Isabelle/RL theory (Isabelle_RL.thy) for loading ML files
  val isabelle_rl_path = Path.of(Directories.isabelle_rl)
  val thy_path = isabelle_rl_path.resolve("src/main/ml/Libraries.thy")
  if (!Files.exists(thy_path)) {
    throw new Exception(s"Libraries.thy not found at ${thy_path} in /src/main/ml/")
  }
  val isabelle_rl_thy : Theory = Theory(thy_path)
  
  override def toString(): String = {
    "Minion(work_dir=" + work_dir + ", logic=" + logic + ", imports_dir=" + imports_dir + ")"
  }
  

  // GENERIC MINION TASKS 
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

  def test_isabelle (ml_code: String = "if true then \"hi Scala\" else \"bye\""): Unit = { 
    val to_print = compileValue[String](ml_code).retrieveNow
    println(to_print)
  }

  def launch_python_script(script_path: Path): Process = {
    val script_dir = script_path.getParent.toString
    val python_cmnd = "python" // depends on your system setup
    val command = Seq(python_cmnd, script_path.toString())
    println(s"Attempting to start Python server: ${command.mkString(" ")}")
    val process: Process = Process(command, new File(script_dir)).run()
    return process
  }

  // MINION WRITING TASKS
  private object ML_writer {
    val ml_writer = isabelle_rl_thy.importMLStructureNow("Writer")
    final val write_json_proofs : MLFunction3[String, Theory, String, Unit] 
      = compileFunction[String, Theory, String, Unit](s"${ml_writer}.write_json_proofs")
  }

  def write_json_proofs (write_dir: Path, thy_file_path: Path): Unit = {
    val thy0 = imports.get_start_theory(thy_file_path)
    val thy_text = imports.get_file_text(thy_file_path)
    ML_writer.write_json_proofs(write_dir.toString(), thy0, thy_text).retrieveNow
  }

  // MINION REPL TASKS
  // TODO: turn into a class and make private?
  object ML_repl {
    private final val ml_repl_struct = isabelle_rl_thy.importMLStructureNow("Repl_State")
    final class Repl_State private (val mlValue: MLValue[Repl_State]) 
      extends MLValueWrapper[Repl_State] {}

    object Repl_State extends MLValueWrapper.Companion[Repl_State] {
      override protected val mlType: String = s"$ml_repl_struct.T" 
      override protected val predefinedException: String = s"$ml_repl_struct.E_Repl_State"
      override protected def instantiate(mlValue: MLValue[Repl_State]): Repl_State = new Repl_State(mlValue)
      
      override protected def newOps(implicit isabelle: Isabelle): Ops = new Ops
      protected class Ops(implicit isabelle: Isabelle) extends super.Ops {
        lazy val size = compileFunction[Repl_State, Int](s"$ml_repl_struct.size")
        lazy val last_state = compileFunction[Repl_State, String](s"$ml_repl_struct.last_state")
        lazy val last_action = compileFunction[Repl_State, String](s"$ml_repl_struct.last_action")
        lazy val last_error = compileFunction[Repl_State, String](s"$ml_repl_struct.last_error")
        lazy val is_at_proof = compileFunction[Repl_State, Boolean](s"$ml_repl_struct.is_at_proof")
        lazy val no_subgoals = compileFunction[Repl_State, Boolean](s"$ml_repl_struct.no_subgoals")
        lazy val proof_so_far = compileFunction[Repl_State, String](s"$ml_repl_struct.proof_so_far")
        lazy val last_proof_of = compileFunction[Repl_State, String](s"$ml_repl_struct.last_proof_of")
        
        lazy val init = compileFunction[Theory, Repl_State](s"$ml_repl_struct.init")
        lazy val read_eval = compileFunction[String, Repl_State, Repl_State](s"$ml_repl_struct.read_eval")
        lazy val print = compileFunction[Repl_State, String](s"$ml_repl_struct.print")
        lazy val undo = compileFunction[Repl_State, Repl_State](s"$ml_repl_struct.undo")
        lazy val reset = compileFunction[Repl_State, Repl_State](s"$ml_repl_struct.reset")
        lazy val go_to = compileFunction[Theory, String, String, Repl_State](s"$ml_repl_struct.go_to")
      }
    }
  }

  // repl state info
  def repl_size (state: ML_repl.Repl_State): Int = 
    ML_repl.Repl_State.Ops.size(state).retrieveNow
  
  def repl_last_usr_st(state: ML_repl.Repl_State): String =
    ML_repl.Repl_State.Ops.last_state(state).retrieveNow
  
  def repl_last_action(state: ML_repl.Repl_State): String =
    ML_repl.Repl_State.Ops.last_action(state).retrieveNow
  
  def repl_last_error (state: ML_repl.Repl_State): String = 
    ML_repl.Repl_State.Ops.last_error(state).retrieveNow
  
  def repl_is_at_proof(state: ML_repl.Repl_State): Boolean =
    ML_repl.Repl_State.Ops.is_at_proof(state).retrieveNow
  
  def repl_without_subgoals(state: ML_repl.Repl_State): Boolean =
    ML_repl.Repl_State.Ops.no_subgoals(state).retrieveNow
  
  def repl_proof_so_far(state: ML_repl.Repl_State): String =
    ML_repl.Repl_State.Ops.proof_so_far(state).retrieveNow

  def repl_last_proof_of(state: ML_repl.Repl_State): String =
    ML_repl.Repl_State.Ops.last_proof_of(state).retrieveNow

  // repl state ops
  def repl_init (thy: Theory): ML_repl.Repl_State = 
    ML_repl.Repl_State.Ops.init(thy).retrieveNow

  def repl_apply (txt: String, state: ML_repl.Repl_State) = 
    ML_repl.Repl_State.Ops.read_eval(txt, state).retrieveNow

  def repl_print (state: ML_repl.Repl_State) = 
    ML_repl.Repl_State.Ops.print(state).retrieveNow

  def repl_undo (state: ML_repl.Repl_State) = 
    ML_repl.Repl_State.Ops.undo(state).retrieveNow

  def repl_reset (state: ML_repl.Repl_State) = 
    ML_repl.Repl_State.Ops.reset(state).retrieveNow

  def repl_go_to (thy: Theory, file_path: String, act_txt: String) = 
    ML_repl.Repl_State.Ops.go_to(thy, file_path, act_txt).retrieveNow
}