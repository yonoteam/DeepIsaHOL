/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Writes proof data from input read directory to output write directory
*/

package isabelle_rl
import java.io.{File, FileWriter, IOException}
import java.nio.file.{Path, Files, Paths}
import scala.util.{Failure, Success, Try}
import scala.concurrent.{Future, Await}
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Try, Success, Failure}
import java.nio.file.FileAlreadyExistsException
import java.util.logging.{Level, Logger, FileHandler, SimpleFormatter}
import scala.jdk.CollectionConverters._
import isabelle_rl.Isa_Minion

class Writer(val read_dir: String, val write_dir: String, val logic: String = "HOL") {
  // write_dir
  Files.createDirectories(Paths.get(write_dir))

  // logger
  private val logger: Logger = Logger.getLogger("ErrorLogger")
  
  val log_file: FileHandler = new FileHandler(s"$write_dir/error.log", true)
  log_file.setFormatter(new SimpleFormatter)
  logger.addHandler(log_file)
  logger.setLevel(Level.SEVERE)

  // minion
  private val minion: Isa_Minion = new Isa_Minion(read_dir, logic)
  println(s"Initialised minion for directory: $read_dir")

  def get_minion(): Isa_Minion = minion

  // format
  private var format: String = Writer.json_format
  private val json = Writer.json_format
  private val g2tac = Writer.g2tac_format

  def set_format(new_format: String): Unit = {
    new_format match {
      case `json` => format = new_format
      case `g2tac` => format = new_format
      case _ =>
        println(s"Undefined writing format $new_format, defaulting to $format")
    }
  }

  // find the .thy file inside the minion's work directory or its subdirectories
  def get_theory_file_path(read_file_name: String): Option[Path] = {
    val read_file_path = Path.of(read_file_name)
    if (Files.exists(read_file_path)) {
      Some(read_file_path)
    } else {
      val proper_file_name = if (read_file_name.endsWith(".thy")) read_file_name else {
        read_file_name + ".thy"
      }
      val path_to_file = minion.imports.local_thy_files.find(_.endsWith(proper_file_name))
      path_to_file
    }
  }

  def get_proofs_data(file_path: Path): List[String] = {
    Try(minion.extract(file_path)) match {
        case Success(json_java_strs) => json_java_strs.map(_.replace("\\<", "\\\\<"))
        case Failure(exception) =>
        logger.severe(s"Failed to extract data from $file_path: ${exception.getMessage}")
        List.empty
    }
  }

  // Write proof data from the corresponding file
  def write_data(file_name: String): Unit = {
    val read_file_path = get_theory_file_path(file_name).get
    format match {
      case `json` =>
        val rel_path = Paths.get(read_dir).relativize(read_file_path).toString
        val target_dir = Paths.get(write_dir, rel_path.toString().stripSuffix(".thy"))
        Files.createDirectories(target_dir)
        Try {
          minion.write_json_proofs(target_dir, read_file_path)
        } match {
          case Failure(exception) =>
            logger.severe(s"Error writing data from $read_file_path: ${exception.getMessage}")
          case Success(_) => ()
        }
      
      case `g2tac` =>
        val target_dir = Path.of(write_dir)
        println(s"Writing to $target_dir")
        Try {
          minion.write_g2tac_proofs(target_dir, read_file_path)
        } match {
          case Failure(exception) =>
            logger.severe(s"Error writing data from $read_file_path: ${exception.getMessage}")
          case Success(_) => ()
        }

      case some_other_format: String => 
        val error_message = s"Unknown writing format: $some_other_format. Skipping $file_name."
        logger.severe(error_message)
        throw new IllegalArgumentException(error_message)
    }
  }

  def write_data_with_timeout(file_name: String, timeout_min: Int): Try[Unit] = {
    val write_cmnd = Future { write_data(file_name) }
    try {
      Await.result(write_cmnd, timeout_min.minutes)
      Success(())
    } catch {
      case _: java.util.concurrent.TimeoutException =>
        println(s"Writing the proof exceeded the timeout of $timeout_min minutes")
        Failure(new Exception(s"Writing the proof exceeded the timeout of $timeout_min minutes"))
      case e: Exception =>
        Failure(e)
    }
  }

  // extract data from all .thy files in the read_dir into the write_dir
  def write_all(): Unit = {
    minion.imports.to_local_list().foreach { path =>
      println(s"Creating proofs for $path")
      write_data_with_timeout(path.toString(), 3)
    }
    println("Done")
  }
}

object Writer {
  val json_format = "JSON"
  val g2tac_format = "G2TAC"
}