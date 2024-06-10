/*  Maintainer: Jonathan Julián Huerta y Munive
    Email: jonjulian23@gmail.com

Isabelle session initialisation.
*/

// package learn_isabelle

import isabelle._
import scala.collection.mutable
import scala.compiletime.ops.boolean

object Isabelle_Session {
  def make_options(build: Boolean): Options = {
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

  def make_store(build: Boolean): Store = {
    if (build) {
      val options = make_options(build)
      val build_engine = Build.Engine(Build.engine_name(options))
      val build_hosts = new mutable.ListBuffer[Build_Cluster.Host].toList
        
      build_engine.build_store(options, build_cluster = build_hosts.nonEmpty)
    }
    else Store(options = make_options(build))
  }

  def make_background(logic:String, store: Store, progress: Progress): Sessions.Background = {
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

  def start(logic: String): Unit = {
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
    } catch {
        case exn: Throwable =>
          error("Throwable error: Failed to execute Isabelle session.")
          sys.exit(Process_Result.RC.failure)
    }
  }
  // start("Main")
}