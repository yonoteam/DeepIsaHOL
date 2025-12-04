/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Main entrypoint
*/

package isabelle_rl

import java.io.{File, RandomAccessFile}
import java.nio.channels.FileLock
import java.nio.file.{Path, Paths, Files}
import java.util.concurrent.{Executors, ConcurrentLinkedQueue}
import java.util.logging.{FileHandler, Formatter, Level, LogRecord, Logger}
import scala.concurrent.{ExecutionContext, Future, Await}
import scala.concurrent.duration.Duration
import scala.io.Source
import scala.util.matching.Regex
import scala.util.{Failure, Success, Try}
import scala.jdk.CollectionConverters._

import isabelle_rl._

object Main {
  val root_rgx: Regex = Utils.root_rgx
  var main_read_dir = ""
  var main_write_dir = ""
  var main_progress_file = ""
  var num_threads = 1

  var main_logger: Logger = _

  def set_params(args: Array[String]): Unit = {
    if (args.length >= 2) {
      main_read_dir = args(0)
      main_write_dir = args(1)
      if (args.length >= 3) {
        num_threads = args(2).toInt
      }
    } else {
      val usage_message = """Usage: 
          1st input - read directory.
          2nd input - write directory.
          3rd input (optional) - number of concurrent writers (default 1)."""
      println(usage_message)
      sys.exit(1)
    }
    main_progress_file = Paths.get(main_write_dir, "progress.txt").toString
  }

  def setup_main_logger(): Unit = {
    val log_path = Paths.get(main_write_dir, "main_writer.log")
    main_logger = Logger.getLogger("Main_Writer_Logger")
    
    // Clear previous handlers
    for (h <- main_logger.getHandlers) main_logger.removeHandler(h)

    val fh = new FileHandler(log_path.toString, true)
    fh.setFormatter(new Formatter {
      override def format(record: LogRecord): String = {
        val date = new java.util.Date(record.getMillis)
        s"[$date] [${record.getLevel}] ${formatMessage(record)}\n"
      }
    })
    main_logger.addHandler(fh)
    main_logger.setLevel(Level.INFO)
    main_logger.setUseParentHandlers(true) // print to stdout to see progress
  }

  def check_params(): Unit = {
    // read directory
    val read_dir = new File(main_read_dir)
    if (!read_dir.exists() || !read_dir.isDirectory()) {
      println(s"Error: read directory $main_read_dir does not exist or is not a directory.")
      sys.exit(1)
    }

    // write directory
    val write_dir = new File(main_write_dir)
    if (!write_dir.exists()) {
      if (write_dir.mkdirs()) {
        println(s"Created write directory $main_write_dir")
      } else {
        println(s"Failed to create write directory $main_write_dir")
        sys.exit(1)
      }
    } else if (!write_dir.isDirectory()) {
      println(s"The input $main_write_dir is not a valid directory.")
      sys.exit(1)
    }
  }

  def get_task_type(top_dir: File): String = {
    val all_files = top_dir.listFiles().filter(_.isFile).toSet
    val file_names = all_files.map(_.getName)

    val root_files = all_files.filter(file => file.getName.startsWith("ROOT")).map(_.getName())
    if (file_names.contains("ROOTS")) "ROOTS"
    else if (file_names.contains("ROOT")) "ROOT"
    else if (file_names.exists(_.endsWith(".thy"))) "THY"
    else {
      val err_mssg = s"Read directory $top_dir does not contain (immediate) ROOT, ROOTS, or .thy files, aborting."
      if (main_logger != null) main_logger.severe(err_mssg)
      else println(err_mssg)
      sys.exit(1)
    }
  }

  // load progress from progress file with shared lock
  def load_progress(): Set[String] = {
    val progress_file = new File(main_progress_file)
    if (progress_file.exists()) {
      val raf = new RandomAccessFile(progress_file, "r")
      val channel = raf.getChannel
      var lock: FileLock = null
      try {
        lock = channel.lock(0, Long.MaxValue, true)
        val is = java.nio.channels.Channels.newInputStream(channel)
        Source.fromInputStream(is).getLines().filter(_.trim.nonEmpty).toSet
      } catch {
        case e: Exception =>
          main_logger.warning(s"Could not load progress: ${e.getMessage}")
          Set.empty[String]
      } finally {
        if (lock != null) lock.release()
        channel.close()
        raf.close()
      }
    } else {
      Set.empty[String]
    }
  }

  // save progress to progress file with exclusive lock
  def save_progress(sub_dir: String): Unit = synchronized{
    val progress_file = new File(main_progress_file)
    val raf = new RandomAccessFile(progress_file, "rw")
    val channel = raf.getChannel
    var lock: FileLock = null
    try {
      lock = channel.lock()
      raf.seek(raf.length())
      raf.writeBytes(sub_dir + "\n")
    } finally {
      if (lock != null) lock.release()
      channel.close()
      raf.close()
    }
  }

  // finds the first logic in the ROOT file
  def find_logic(root_file: File): Option[String] = {
    val root_src = Source.fromFile(root_file)
    try {
      val content = root_src.mkString
      root_rgx.findFirstMatchIn(content) match {
        case Some(m) => 
          println(s"Found logic = ${m.group(1)}")
          Some(m.group(1))
        case _ => None
      }
    } finally {
      root_src.close()
    }
  }

  def launch_writer(read_dir: String, write_dir: String, logic: String)(implicit ec: ExecutionContext): Unit = {
    val writer_Try = Try {
      new Writer(read_dir, write_dir, logic)
    }

    val result = writer_Try.flatMap { writer =>
      Try {
        writer.write_all()
      }.transform(
        res => {
          writer.shutdown_isabelle()
          Success(res)
        },
        exception => {
          writer.shutdown_isabelle()
          Failure(exception)
        }
      )
    }

    result.failed.foreach { exception =>
      main_logger.severe(s"Error writing data from $read_dir to $write_dir:\n ${exception.getMessage}")
    }
  }

  def do_roots_task (top_read_dir: File, top_write_dir: File): Unit = {
    val processed = load_progress()
    val sub_dirs = top_read_dir.listFiles().filter(_.isDirectory)

    val to_process = sub_dirs.filterNot(d => processed.contains(d.getName)).toList
    if (to_process.isEmpty) {
      main_logger.info("All subdirectories already processed.")
      return
    }

    main_logger.info(s"Found ${sub_dirs.length} directories. ${processed.size} processed. ${to_process.size} remaining.")
    val workQueue = new ConcurrentLinkedQueue[File](to_process.asJava)

    val executor = Executors.newFixedThreadPool(num_threads)
    implicit val ec = ExecutionContext.fromExecutorService(executor)

    val workerFutures = (1 to num_threads).map { num =>
      Future {
        var active = true
        while (active) {
          val sub_dir = workQueue.poll()
          if (sub_dir == null) { active = false }
          else {
            main_logger.info(s"[Thread-$num] picked up: ${sub_dir.getName}")
            val root_file = new File(sub_dir, "ROOT")
            if (!root_file.exists()) {
              main_logger.warning(s"[Thread-$num] No ROOT file found in ${sub_dir.getName}")
            } else {
              find_logic(root_file) match {
                case Some(logic) =>
                  val read_dir = sub_dir.getAbsolutePath()
                  val write_dir = new File(top_write_dir, s"${sub_dir.getName}").getAbsolutePath()
                  launch_writer(read_dir, write_dir, logic)
                  save_progress(sub_dir.getName)
                  main_logger.info(s"[Thread-$num] Processed: ${sub_dir.getName}")
                case None =>
                  main_logger.warning(s"[Thread-$num] No logic found in ROOT file for ${sub_dir.getName}")
              }
            }
          }
        }
      }
    }
    try {
      Await.result(Future.sequence(workerFutures), Duration.Inf)
      main_logger.info("Finished all writing tasks.")
    } finally {
      executor.shutdown()
    }
  }

  def main (args: Array[String]): Unit = {
    set_params(args)
    check_params()
    setup_main_logger()
    main_logger.info(s"Params checked. Read: $main_read_dir, Write: $main_write_dir, Threads: $num_threads")

    val top_read_dir = new File(main_read_dir)
    val top_write_dir = new File(main_write_dir)
    val task = get_task_type(top_read_dir)
    main_logger.info(s"Writing task based on detected file: $task")

    implicit val ec: ExecutionContext = ExecutionContext.global // do_roots_task handles its own ec

    task match {
      case "ROOTS" => do_roots_task(top_read_dir, top_write_dir)

      case "ROOT" =>
        val root_file = new File(top_read_dir, "ROOT")
        find_logic(root_file) match {
          case Some(logic) =>
            val read_dir = top_read_dir.getAbsolutePath()
            val write_dir = top_write_dir.getAbsolutePath()
            launch_writer(read_dir, write_dir, logic)
          case None =>
            main_logger.severe(s"No logic found in ROOT file ${root_file.getAbsolutePath()}")
        }
      
      case "THY" => 
        val read_dir = top_read_dir.getAbsolutePath()
        val write_dir = top_write_dir.getAbsolutePath()
        val logic = if (args.length > 3) args(3) else "HOL"
        launch_writer(read_dir, write_dir, args(3))
    }
  }
}

