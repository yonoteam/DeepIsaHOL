/*  Maintainer: Jonathan Julián Huerta y Munive

Isabelle session initialisation.
*/

// package learn_isabelle

import isabelle._
import scala.collection.mutable
import scala.compiletime.ops.boolean
import javax.print.Doc
import javax.swing.text.Document

object Isabelle_Run {
  private def make_options(build: Boolean): Options = {
    val options0 = Options.init0()

    val options1 = 
      if (build) options0 ++ Options.Spec.ISABELLE_BUILD_OPTIONS 
      else options0

    val options2 =
      Isabelle_System.getenv("JEDIT_BUILD_MODE") match {
        case "default" => options1
        case mode => options1.bool.update("system_heaps", mode == "system")
      }

    val options3 =
      Isabelle_System.getenv("JEDIT_PROCESS_POLICY") match {
        case "" => options2
        case s => options2.string.update("process_policy", s)
      }

    options3
  }

  private def make_store(build: Boolean): Store = {
    if (build) {
      val options = make_options(build)
      val build_engine = Build.Engine(Build.engine_name(options))
      val build_hosts = new mutable.ListBuffer[Build_Cluster.Host].toList
        
      build_engine.build_store(options, build_cluster = build_hosts.nonEmpty)
    }
    else Store(options = make_options(build))
  }

  private def make_background(logic:String, store: Store, progress: Progress): Sessions.Background = {
    try {
      val afp_root = Some(AFP.BASE) // FIXME: requires AFP installed as component

      val dirs = new mutable.ListBuffer[Path].toList

      val include_sessions: List[String] = 
        space_explode(':', Isabelle_System.getenv("JEDIT_INCLUDE_SESSIONS"))

      Sessions.background(store.options, 
        logic, 
        progress = progress, 
        dirs = AFP.main_dirs(afp_root) ::: dirs, 
        include_sessions = include_sessions
      ).check_errors
    } catch {
        case exn: Throwable =>
          error("Throwable error: Failed to make background.")
    }
  }

  def apply(logic: String): Isabelle_Run = {
    try {
      Isabelle_System.init()
      
      val progress = new Console_Progress()
          
      val store = make_store(build=false)

      val session_background = make_background(logic, store, progress)

      // FIXME: requires existing heap image for logic
      val session_heaps =
        ML_Process.session_heaps(store, session_background, logic = session_background.session_name)

      val log: Logger = Logger.make_system_log(progress, store.options)

      val resources = new Resources(session_background, log)

      val session = new Session(store.options, resources)

      progress.echo("Starting session " + logic + " ...")

      val process = Isabelle_Process.start(
        store.options, session, session_background, session_heaps)//.await_startup()

      new Isabelle_Run(session, process, progress)
    } catch {
        case exn: Throwable =>
          error("Throwable error: Failed to launch Isabelle process.")
          sys.exit(Process_Result.RC.failure)
    }
  }
  
  def test_edits(): Session.Raw_Edits = {
    val test_blobs = isabelle.Document.Blobs.make(List())
    val node_name = isabelle.Document.Node.Name("Draft.Isa_Scala_Basics","Draft.Isa_Scala_Basics")
    val header_name = isabelle.Document.Node.Name("Draft.Isa_Scala_Basics","HOL.HOL")
    val header1 = isabelle.Document.Node.Header(List((header_name, List(("offset","35"), ("end_offset","42"), ("id","command")))))
    val deps1: isabelle.Document.Node.Deps[isabelle.Text.Edit, isabelle.Text.Perspective] = isabelle.Document.Node.Deps(header1)
    val edit1 = (node_name, deps1)
    val test_str = """
theory Isa_Scala_Basics
  imports HOL.HOL
begin
lemma my_lemma: "True"
  by simp
end"""
    val text_edit2: isabelle.Document.Node.Edits[isabelle.Text.Edit, isabelle.Text.Perspective] = isabelle.Document.Node.Edits(List(Text.Edit.insert(0,test_str)))
    val edit2 = (node_name, text_edit2)
    val perspective3: isabelle.Document.Node.Perspective[isabelle.Text.Edit, isabelle.Text.Perspective] = isabelle.Document.Node.Perspective(false, Text.Perspective(List(Text.Range(0,197))), isabelle.Document.Node.Overlays.empty)
    val edit3 = (node_name, perspective3)
    val raw_edits = new Session.Raw_Edits(test_blobs, List(edit1, edit2, edit3))
    raw_edits
  }
}

class Isabelle_Run(val session: Session, val process: Isabelle_Process, val progress: Progress) {
  def send_edits: Unit = {
    val raw_edits = Isabelle_Run.test_edits()
    session.manager.send(raw_edits)
  }

  def stop(): Unit = {
    session.stop()
    process.terminate()
  }
}