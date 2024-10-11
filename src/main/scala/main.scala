/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Main entrypoint
*/

package isabelle_rl

import java.io.{File, FileOutputStream, PrintWriter, BufferedWriter}
import scala.io.Source
import scala.util.matching.Regex
import scala.util.{Failure, Success, Try}

import isabelle_rl.Directories
import isabelle_rl._


object Main {
  val root_rgx: Regex = """session\s+"?([\w-]+)"?\s+(in\s+"?[\w\/-]+"?)?\s*=""".r

  // Finds the logic in the ROOT file
  def find_logic(root_file: File): Option[String] = {
    val root_src = Source.fromFile(root_file)
    try {
      val content = root_src.mkString
      root_rgx.findFirstMatchIn(content) match {
        case Some(m) => 
          println(s"root match = ${m.group(1)}")
          Some(m.group(1))
        case _ => None
      }
    } finally {
      root_src.close()
    }
  }

  // Load progress from file
  def load_progress(): Set[String] = {
    val progress_file = new File(Directories.progress_file)
    if (progress_file.exists()) {
      Source.fromFile(progress_file).getLines().toSet
    } else {
      Set.empty[String]
    }
  }

  // Save progress to file
  def save_progress(sub_dir: String): Unit = {
    val writer = new BufferedWriter(new PrintWriter(new FileOutputStream(new File(Directories.progress_file), true)))
    try {
      writer.write(sub_dir + "\n")
    } finally {
      writer.close()
    }
  }

  def main (args: Array[String]): Unit = {
    val afp_dir = new File(Directories.isabelle_afp)
    val processed = load_progress()

    afp_dir.listFiles().filter(_.isDirectory).foreach { sub_dir =>
      if (! processed.contains(sub_dir.getName)) {
        val root_file = new File(sub_dir, "ROOT")
        if (root_file.exists()) {
          find_logic(root_file) match {
            case Some(logic) =>
              val read_dir = s"${Directories.isabelle_afp}" + s"${sub_dir.getName}"
              val write_dir = s"${Directories.test_write_dir}" + s"${sub_dir.getName}"
              Try {
                println(s"\nInitialising writer with read_dir = ${read_dir} \nand write_dir ${write_dir}")
                val writer = new Writer(read_dir, write_dir, logic)
                val minion = writer.get_minion()
                implicit val isabelle:de.unruh.isabelle.control.Isabelle = minion.isabelle
                writer.write_all()
                isabelle.destroy()
              } match {
                case Failure(exception) =>
                  println(s"Error starting writer for $read_dir: ${exception.getMessage}")
                case Success(_) => ()
              }
              save_progress(sub_dir.getName)
              println(s"Processed: ${sub_dir.getName}\n")
            case None =>
              println(s"No logic found in ROOT file for ${sub_dir.getName}")
          }
        } else {
          println(s"No ROOT file found in ${sub_dir.getName}")
        }
      } else {
        println(s"Skipping already processed sub_dir: ${sub_dir.getName}")
      }
    }
  }
}

