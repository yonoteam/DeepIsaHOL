/*  
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Main entrypoint
*/

package isabelle_rl

import java.io.{File, FileOutputStream, PrintWriter, BufferedWriter}
import java.nio.file.{Path, Files, Paths}
import scala.io.Source
import scala.util.matching.Regex
import scala.util.{Failure, Success, Try}

import isabelle_rl._

object Main {
  val root_rgx: Regex = """session\s+"?([\w-]+)"?\s+(in\s+"?[\w\/-]+"?)?\s*=""".r
  var main_write_format = ""
  var main_write_dir = ""
  var main_progress_file = ""
  var main_read_dir = ""

  def set_params(args: Array[String]): Unit = {
    args.length match {
      case 3 =>
        main_write_format = args(0)
        main_write_dir = args(1)
        main_read_dir = args(2)
      case 4 =>
        main_write_format = args(0)
        main_write_dir = args(1)
        main_read_dir = args(2)
      case _: Int =>
        val usage_message = """Usage: 
          1st input - write format. Options JSON and G2TAC.
          2nd input - write directory.
          3rd input - read directory.
          4th input - logic. Not used if there is a ROOT file in the read directory. Default otherwise is HOL."""
        println(usage_message)
        sys.exit(1)
        return
    }
    main_progress_file = main_write_dir + "/progress.txt"
  }

  def check_params(): Unit = {
    // read directory
    val read_dir = new File(main_read_dir)
    if (!read_dir.exists() || !read_dir.isDirectory()) {
      println(s"Read directory $main_read_dir does not exist or is not a directory.")
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

    // write format
    val formats = Set(Writer.json_format, Writer.g2tac_format)
    if (! formats(main_write_format)) {
      println(s"Undefined writing format $main_write_format. Options are: ${formats.toString()}")
      sys.exit(1)
    }
  }

  def get_task_type(top_dir: File): String = {
    val all_files = top_dir.listFiles().filter(file => file.isFile).toSet
    val root_files = all_files.filter(file => file.getName.startsWith("ROOT")).map(_.getName())
    if (root_files("ROOTS")) {
      return "ROOTS"
    } else if (root_files("ROOT")) {
      return "ROOT"
    } else if (all_files.exists(_.toPath.endsWith(".thy"))) {
      return "THY"
    } else {
      println(s"Write directory $main_write_dir does not contain ROOT or .thy files, aborting.")
      sys.exit(1)
    }
  }

  // load progress from progress file
  def load_progress(): Set[String] = {
    val progress_file = new File(main_progress_file)
    if (progress_file.exists()) {
      Source.fromFile(progress_file).getLines().toSet
    } else {
      Set.empty[String]
    }
  }

  // save progress to progress file
  def save_progress(sub_dir: String): Unit = {
    val writer = new BufferedWriter(
      new PrintWriter(
        new FileOutputStream(
          new File(main_progress_file), true
    )))
    try {
      writer.write(sub_dir + "\n")
    } finally {
      writer.close()
    }
  }

  // finds the logic in the ROOT file
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

  def write_all(read_dir: String, write_dir: String, logic: String): Unit = {
    Try {
      println(s"\nInitialising writer with read_dir = ${read_dir} \nand write_dir ${write_dir}")
      val writer = new Writer(read_dir, write_dir, logic)
      writer.set_format(main_write_format)
      val minion = writer.get_minion()
      implicit val isabelle:de.unruh.isabelle.control.Isabelle = minion.isabelle
      writer.write_all()
      isabelle.destroy()
    } match {
      case Failure(exception) =>
        println(s"Error writing data from $read_dir to $write_dir:\n ${exception.getMessage}")
      case Success(_) => ()
    }     
  }

  def do_roots_task (top_read_dir: File, top_write_dir: File): Unit = {
    val processed = load_progress()
    top_read_dir.listFiles().filter(_.isDirectory).foreach { sub_dir =>
      // if the sub_dir has not been processed according to progress file
      if (! processed.contains(sub_dir.getName)) {
        val root_file = new File(sub_dir, "ROOT")
        if (root_file.exists()) {
          find_logic(root_file) match {
            case Some(logic) =>
              val read_dir = sub_dir.getAbsolutePath()
              val write_dir = new File(top_write_dir, s"${sub_dir.getName}").getAbsolutePath()
              write_all(read_dir, write_dir, logic)
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

  def main (args: Array[String]): Unit = {
    set_params(args)
    check_params()

    val top_read_dir = new File(main_read_dir)
    val top_write_dir = new File(main_write_dir)
    val task = get_task_type(top_read_dir)

    task match {
      case "ROOTS" => do_roots_task(top_read_dir, top_write_dir)

      case "ROOT" =>
        val root_file = new File(top_read_dir, "ROOT")
        find_logic(root_file) match {
          case Some(logic) =>
            val read_dir = top_read_dir.getAbsolutePath()
            val write_dir = top_write_dir.getAbsolutePath()
            write_all(read_dir, write_dir, logic)
          case None =>
            println(s"No logic found in ROOT file ${root_file.getAbsolutePath()}")
        }
      
      case "THY" => 
        val read_dir = top_read_dir.getAbsolutePath()
        val write_dir = top_write_dir.getAbsolutePath()
        write_all(read_dir, write_dir, args(3))
    }
  }
}

