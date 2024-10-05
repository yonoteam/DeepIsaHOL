/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Writes proof data from input read directory to output write directory
*/

package isabelle_rl
import java.io.{File, FileWriter, IOException}
import java.nio.file.{Path, Files, Paths}
import scala.util.{Failure, Success, Try}
import java.nio.file.FileAlreadyExistsException
import java.util.logging.{Level, Logger, FileHandler, SimpleFormatter}
import scala.jdk.CollectionConverters._
import isabelle_rl.Isa_Minion

class Writer(read_dir: String, write_dir: String, logic: String = "HOL") {

  // write_dir
  Files.createDirectories(Paths.get(write_dir))

  // logger
  private val logger: Logger = Logger.getLogger("ErrorLogger")
  
  val log_file: FileHandler = new FileHandler(s"$write_dir/error.log", true)
  log_file.setFormatter(new SimpleFormatter)
  logger.addHandler(log_file)
  logger.setLevel(Level.SEVERE)

  // minion
  private val minion: Isa_Minion = Isa_Minion(logic, read_dir)
  println(s"Initialized minion for directory: $read_dir")

  def get_minion(): Isa_Minion = minion

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
    //val data_list = get_proofs_data(read_file_path)
    val rel_path = Paths.get(read_dir).relativize(read_file_path).toString
    val target_dir = Paths.get(write_dir, rel_path.toString().stripSuffix(".thy"))
    Files.createDirectories(target_dir)
    Try {
      minion.write_proofs(target_dir, read_file_path)
    } match {
      case Failure(exception) =>
        logger.severe(s"Error writing data from $read_file_path: ${exception.getMessage}")
      case Success(_) => ()
    }
    // data_list.zipWithIndex.foreach { case (data, counter) =>
    //   val json_name = s"proof$counter.json"
    //   val target_file_path = target_dir.resolve(json_name)
    //   if (Files.exists(target_file_path)) {
    //     val message = s"File $target_file_path already exists. Aborting to avoid overwriting."
    //     logger.severe(message)
    //      throw new FileAlreadyExistsException(message)
    //   } else {
    //     Try {
    //       val writer = new FileWriter(target_file_path.toFile)
    //       try {writer.write(data)} 
    //       finally {writer.close()}
    //     } match {
    //       case Failure(exception) =>
    //         logger.severe(s"Error writing file $target_file_path: ${exception.getMessage}")
    //       case Success(_) => ()
    //     }
    //   }
    // }
  }

  // extract data from all .thy files in the read_dir into the write_dir
  def write_all(): Unit = {
    minion.imports.local_thy_files.foreach { path =>
      println(s"Creating proofs for $path")
      write_data(path.toString())
    }
    println("Done")
  }
}