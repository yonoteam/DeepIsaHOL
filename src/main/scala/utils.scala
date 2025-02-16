/*
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Useful functions for user-setup information
*/

package isabelle_rl

import java.io.{File, FileOutputStream, PrintWriter, BufferedWriter}
import java.nio.file.{Path, Paths, Files}

import scala.util.matching.Regex
import scala.io.Source
import scala.collection.mutable

import scala.jdk.StreamConverters._

import isabelle_rl.Graph


object Utils {
  private val AFP_path = Paths.get(Directories.isabelle_afp)
  private val AFP_ROOTS = new File(AFP_path.resolve("ROOTS").toString())
  private val isa_app_path = Paths.get(Directories.isabelle_afp)
  private val isa_app_ROOTS = new File(isa_app_path.resolve("ROOTS").toString())
  private val debug = false

  def valid_afp: Boolean = AFP_ROOTS.exists() && AFP_ROOTS.isFile // TODO: add more checks

  def is_isa_root_dir(dir: Path): Boolean = {
    val root_file = dir.resolve("ROOT")
    Files.exists(root_file) && Files.isRegularFile(root_file)
  }

  // Warning: functions finding logic-related info in this file depend on the correctness of this regex
  val root_rgx: Regex = """session\s+"?([\w+-]+)"?\s*(\(\s*([^"]+)\s*\))?\s*(in\s+"?([\w\/-]+)"?)?\s*=""".r

  // tests that the root_rgx finds a logic in each root file
  def test_root_rgx(dir_with_root_files: Path): Unit = {
    val root_files = Files.walk(dir_with_root_files).toScala(Seq)
      .filter(path => Files.isRegularFile(path) && path.getFileName.toString == "ROOT")

    root_files.foreach { root_file =>
      println(s"Searching in file: $root_file")
      val content = Source.fromFile(root_file.toFile).mkString

      // Find matches using root_rgx and print them
      val matches = root_rgx.findAllMatchIn(content).toList
      if (matches.nonEmpty) {
        matches.foreach { m =>
          val session_name = m.group(1)
          val location = Option(m.group(3)).orElse(Option(m.group(4)))
          val none_msg = s"not explicit in ROOT, assuming ${root_file.getParent()}"
          println(s"Session: $session_name, Location ${location.getOrElse(none_msg)}")
        }
      } else {
        println(s"No matches found.")
      }
    }
  }

  // returns a map with the path of all logics in a ROOT file 
  def get_root_logic_paths(root_file: File): Map[String, File] = {
    var result: Map[String, File] = Map()
    val parent_dir = root_file.getParentFile()
    val root_src = Source.fromFile(root_file)
    try {
      val content = root_src.mkString
      root_rgx.findAllMatchIn(content).foreach { m =>
        val logic = m.group(1)
        val logic_path = Option(m.group(3)).orElse(Option(m.group(4))) match {
          case None => parent_dir
          case Some(location) => 
            val possible_loc = new File(parent_dir, location)
            if (possible_loc.isDirectory()) { possible_loc } else { parent_dir }
        }
        result += (logic -> logic_path)
      }
    } finally {
      root_src.close()
    }
    result
  }

  // returns a map with the path of all logics in the ROOTS file
  def get_roots_logic_paths(roots_dir: String): Map[String, Path] = {
    val logics_map = mutable.Map[String, Path]()
    val roots_file = new File(roots_dir)
    if (roots_file.exists && roots_file.isFile) {
      val roots_content = Source.fromFile(roots_file)
      
      try {
        for (logic_path <- roots_content.getLines()) {
          val logic_dir = roots_file.toPath.getParent.resolve(logic_path)
          val curr_logic = logic_dir.getFileName().toString()
          val root_file = logic_dir.resolve("ROOT").toFile

          if (Files.exists(logic_dir) && Files.isDirectory(logic_dir)) {
            if (root_file.exists()) {
              val root_logics = get_root_logic_paths(root_file)

              root_logics.get(curr_logic) match {
                case Some(logic_dir) =>
                  logics_map += (curr_logic -> logic_dir.toPath)
                case None =>
                  if (debug) println(s"Logic $curr_logic not found in ${root_file.getPath}")
              }
            } else {
              if (debug) println(s"ROOT file not found in directory: ${logic_dir}")
            }
          } else {
            if (debug) println(s"Directory for logic $curr_logic does not exist at path: ${logic_dir}")
          }
        }
      } finally {
        roots_content.close()
      }
    } else {
      println(s"The input ${roots_dir} is not a path to a ROOTS directory.")
    }

    logics_map.toMap
  }

  // returns a map with the path of all afp and Isabelle libraries logics
  val logics_map: Map[String, Path] = {
    val afp_logic_paths = get_roots_logic_paths(AFP_ROOTS.toString())
    val isa_app_logic_paths = get_roots_logic_paths(isa_app_ROOTS.toString())
    afp_logic_paths ++ isa_app_logic_paths
  }

  val known_logics = logics_map.keys.toArray.toList

  def get_logic_path (logic:String): Option[Path] = {
    logics_map.get(logic)
  }

}
