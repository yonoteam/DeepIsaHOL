/*  Maintainer: Jonathan Julián Huerta y Munive
    Email: jonjulian23@gmail.com

Isabelle session initialisation.
*/

// package learn_isabelle

import isabelle._
import scala.collection.mutable

object Isabelle_Session {
  def start(session_name: String): Unit = {
    try {
      Isabelle_System.init()
      
      // creating the store 
      val progress = new Console_Progress()
    
      val options = Options.init(specs = Options.Spec.ISABELLE_BUILD_OPTIONS)
    
      val build_engine = Build.Engine(Build.engine_name(options))
    
      val build_hosts = new mutable.ListBuffer[Build_Cluster.Host].toList
    
      val cache = Term.Cache.make()
    
      val store = build_engine.build_store(options, build_cluster = build_hosts.nonEmpty, cache = cache)
      
      // creating the sessions for the context
      val build_options = store.options
    
      val afp_root = Some(AFP.BASE) // FIXME: requires AFP installed as component
    
      val dirs = new mutable.ListBuffer[Path].toList
    
      val selection = Sessions.Selection(base_sessions = new mutable.ListBuffer[String].toList, sessions=List(session_name))
      
      using(store.open_server()) { server =>
        
        val database_server = store.maybe_open_database_server(server = server)
        
        val full_sessions = Sessions.load_structure(build_options, dirs = AFP.main_dirs(afp_root) ::: dirs)
                
        val full_sessions_selection = full_sessions.imports_selection(selection)

        val build_deps = {
            val deps0 = Sessions.deps(full_sessions.selection(selection), progress = progress, inlined_files = true).check_errors
            deps0
        }

        val max_jobs: Option[Int] = None

        val build_context =
            Build.Context(store, build_deps, engine = build_engine, afp_root = afp_root,
                build_hosts = build_hosts, hostname = Build.hostname(build_options),
                store_heap = true, jobs = max_jobs.getOrElse(if (build_hosts.nonEmpty) 0 else 1), master = true)
        
        val sessions_structure = build_context.sessions_structure

        val session_background = build_deps.background(session_name)

        val log: Logger = Logger.make_system_log(progress, build_context.store.options)

        val resources = Headless.Resources(options, session_background, log = log)
        
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

        val session = new Session(options, resources) {
          override val cache: Term.Cache = build_context.store.cache

          override def build_blobs_info(node_name: Document.Node.Name): Command.Blobs_Info =
            Command.Blobs_Info.make(session_blobs(node_name))

          override def build_blobs(node_name: Document.Node.Name): Document.Blobs =
            Document.Blobs.make(session_blobs(node_name))
        }

        val session_heaps = ML_Process.session_heaps(build_context.store, session_background, logic = "Pure") // FIXME: requires existing heap image for logic

        progress.echo("Starting session " + "Pure" + " ...")

        Isabelle_Process.start(
          options, session, session_background, session_heaps).await_startup()
      }
    } catch {
        case exn: Throwable =>
          error("Throwable error: Failed to execute Isabelle session.")
          sys.exit(Process_Result.RC.failure)
        case exn: Server.Error => 
          error("Server error: Failed to execute Isabelle session.")
          error(exn.message)
          sys.exit(Process_Result.RC.failure)
    }
  }
  // start("Main")
}