/*  Maintainer: Jonathan Julián Huerta y Munive
    Email: jonjulian23@gmail.com

Isabelle session initialisation.
*/

// package learn_isabelle

import isabelle._
import scala.collection.mutable
import scala.compiletime.ops.boolean

object Isabelle_Session {
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

      val include_sessions: List[String] = space_explode(':', Isabelle_System.getenv("JEDIT_INCLUDE_SESSIONS"))

      Sessions.background(store.options, logic, dirs = dirs, 
        include_sessions = include_sessions, progress = progress).check_errors
    } catch {
        case exn: Throwable =>
          error("Throwable error: Failed to make session object.")
    }
  }

  def start(logic: String): Unit = {
    try {
      Isabelle_System.init()
      
      val progress = new Console_Progress()
          
      val store = make_store(build=false)

      val session_background = make_background(logic, store, progress)

      val log: Logger = Logger.make_system_log(progress, store.options)

      val resources = new Resources(session_background, log)

      val session = new Session(store.options, resources)
      
      using(store.open_server()) { server =>
        
        val database_server = store.maybe_open_database_server(server = server)
        
        val full_sessions = Sessions.load_structure(store.options, dirs = AFP.main_dirs(afp_root) ::: dirs)
                
        val full_sessions_selection = full_sessions.imports_selection(selection)

        val build_deps = {
            val deps0 = Sessions.deps(full_sessions.selection(selection), progress = progress, inlined_files = true).check_errors
            deps0
        }

        val max_jobs: Option[Int] = None
        
        val session_background = build_deps.background(logic)

        val resources = Headless.Resources(store.options, session_background, log = log)
        
        val session_sources =
            Store.Sources.load(session_background.base, cache = store.cache.compress)

        def session_blobs(node_name: Document.Node.Name): List[(Command.Blob, Document.Blobs.Item)] =
          session_background.base.theory_load_commands.get(node_name.theory) match {
            case None => Nil
            case Some(spans) =>
              val syntax = session_background.base.theory_syntax(node_name)
              val master_dir = Path.explode(node_name.master_dir)
              for (span <- spans; file <- span.loaded_files(syntax).files)
                yield {
                  val src_path = Path.explode(file)
                  val blob_name = Document.Node.Name(File.symbolic_path(master_dir + src_path))

                  val bytes = session_sources(blob_name.node).bytes
                  val text = bytes.text
                  val chunk = Symbol.Text_Chunk(text)

                  Command.Blob(blob_name, src_path, Some((SHA1.digest(bytes), chunk))) ->
                    Document.Blobs.Item(bytes, text, chunk, changed = false)
                }
          }

        val session = new Session(store.options, resources) {
          override val cache: Term.Cache = store.cache

          override def build_blobs_info(node_name: Document.Node.Name): Command.Blobs_Info =
            Command.Blobs_Info.make(session_blobs(node_name))

          override def build_blobs(node_name: Document.Node.Name): Document.Blobs =
            Document.Blobs.make(session_blobs(node_name))
        }

        val session_heaps = ML_Process.session_heaps(store, session_background, logic = "Pure") // FIXME: requires existing heap image for logic

        progress.echo("Starting session " + "Pure" + " ...")

        Isabelle_Process.start(
          store.options, session, session_background, session_heaps).await_startup()
      }
    } catch {
        case exn: Server.Error => 
          error("Server error: Failed to execute Isabelle session.")
          error(exn.message)
          sys.exit(Process_Result.RC.failure)
        case exn: Throwable =>
          error("Throwable error: Failed to execute Isabelle session.")
          sys.exit(Process_Result.RC.failure)
    }
  }
  // start("Main")
}