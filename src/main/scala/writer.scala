/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Writes proof data from input read directory to output write directory
*/

package isabelle_rl
import java.nio.file.{Files, Path, Paths}
import scala.util.{Try, Success, Failure}
import java.util.logging.{FileHandler, Level, LogRecord, Logger, SimpleFormatter}
import scala.concurrent.duration._
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.ExecutionContext.Implicits.global

import isabelle_rl.Isa_Minion

class Writer(val read_dir: String, val write_dir: String, val logic: String = "HOL") {
  // write_dir
  val write_path: Path = Paths.get(write_dir)
  Files.createDirectories(write_path)

  // logger
  private val logger_name = s"Writer_${write_path.toAbsolutePath.toString.hashCode}"
  private val logger: Logger = Logger.getLogger(logger_name)

  for (h <- logger.getHandlers) logger.removeHandler(h) // clean up existing handlers
  private val log_file = new FileHandler(write_path.resolve("processing.log").toString, true)
  log_file.setFormatter(new SimpleFormatter{
    override def format(record: LogRecord): String = {
      s"[${record.getLevel}] ${formatMessage(record)}\n"
    }
  })
  logger.addHandler(log_file)
  logger.setLevel(Level.INFO)
  logger.setUseParentHandlers(false) // not to console

  // minion
  private val minion: Isa_Minion = new Isa_Minion(read_dir, logic, read_dir)
  logger.info(s"Initialised minion for directory: $read_dir")

  def get_minion(): Isa_Minion = minion

  // Write proof data from the corresponding file
  def write_data(file_name: String): Unit = {
    Try {
      val read_file_path = minion.get_theory_file_path(file_name).get
      val rel_path = Paths.get(read_dir).relativize(read_file_path).toString
      val target_dir = Paths.get(write_dir, rel_path.stripSuffix(".thy"))

      Files.createDirectories(target_dir)
      logger.info(s"Writing proofs for: $file_name -> $target_dir")
      minion.write_json_proofs(target_dir, read_file_path)
    } match {
      case Failure(exception) =>
        logger.severe(s"Error writing data from $file_name: ${exception.getMessage}")
      case Success(_) => ()
    }
  }

  def write_data_with_timeout(file_name: String, timeout_min: Int): Try[Unit] = {
    val write_cmnd = Future { write_data(file_name) }
    try {
      Await.result(write_cmnd, timeout_min.minutes)
      Success(())
    } catch {
      case _: java.util.concurrent.TimeoutException =>
        val msg = s"Writing the proof exceeded the timeout of $timeout_min minutes"
        logger.severe(msg)
        Failure(new Exception(msg))
      case e: Exception =>
        logger.severe(s"Exception in write_data_with_timeout: ${e.getMessage}")
        Failure(e)
    }
  }

  // extract data from all .thy files in the read_dir into the write_dir
  def write_all()(implicit ec: ExecutionContext): Unit = {
    minion.imports.to_local_list().foreach { path =>
      logger.info(s"Creating proofs for $path")
      write_data_with_timeout(path.toString(), 3)
    }
    logger.info("Done")
  }

  def shutdown_isabelle(): Unit = {
    if (minion != null && minion.isabelle != null) {
      minion.isabelle.destroy()
      logger.info("Isabelle process destroyed.")
    }
    log_file.close()
  }

  def isabelle_exists(): Boolean = {
    minion != null && minion.isabelle != null && !minion.isabelle.isDestroyed
  }
}