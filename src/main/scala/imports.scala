/*
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Manages the theory-dependency of a given work directory 
 */

package isabelle_rl

import scala.util.matching.Regex
import java.io.{File, FileOutputStream, PrintWriter, BufferedWriter}
import scala.io.Source
import java.nio.file.{Path, Paths, Files}
import java.io.FileNotFoundException
import scala.jdk.CollectionConverters._
import scala.jdk.StreamConverters._
import scala.concurrent.{Future, Await}
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Try, Success, Failure}

import de.unruh.isabelle.control.{Isabelle, OperationCollection}
import de.unruh.isabelle.mlvalue.MLValue.{compileFunction, compileFunction0}
import de.unruh.isabelle.pure.{Position, Theory, TheoryHeader, ToplevelState}
import de.unruh.isabelle.mlvalue.{AdHocConverter, MLFunction, MLFunction0, MLFunction2, MLFunction3}
import isabelle_rl.Graph

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._

object Transition extends AdHocConverter("Toplevel.transition")

class Imports (val work_dir: Path)(implicit isabelle: Isabelle) {
  private val debug = true

  override def toString(): String = {
    "Imports(work_dir=" + work_dir + ")"
  }

  val local_thy_files: List[Path] = {
    val files = Files.walk(work_dir).iterator().asScala
    val filtered_files = files.filter { path => Files.isRegularFile(path) && path.toString.endsWith(".thy")}
    filtered_files.toList
  }

  def get_file_text(path: Path): String = {
    if (Files.isRegularFile(path) && path.toString.endsWith(".thy")) {
      return Files.readString(path)
    } else throw new Exception(s"Imports.get_file_text: input $path is not a .thy file")
  }

  def get_import_names(path: Path): List[String] = {
    val content = get_file_text(path)
    return Imports.Ops.get_header(content, Position.none).retrieveNow.imports
  }

  private def file_name_without_extension(file_path: Path): String = {
    val file_name = file_path.getFileName.toString
    if (file_name.contains(".")) {
      file_name.substring(0, file_name.lastIndexOf('.'))
    } else {
      file_name
    }
  }

  private var dep_graph: Graph[Path, Option[Theory]] = Graph.empty

  def locate_via_thy(import_name: String): Option[(Path,String)] = {
    if (Imports.Ops.can_get_thy(import_name).retrieveNow) {
        Some(Path.of("ISABELLE=" + import_name), "THY=" + import_name)
    } else None
  }

  def locate_locally(import_name: String): Option[(Path,String)] = {
    val file_name = if (import_name.contains(".") && !import_name.contains("/")) {
      Imports.Ops.get_base_name(import_name).retrieveNow + ".thy"
    } else if (import_name.contains("/")) {
      Path.of(import_name + ".thy").getFileName.toString
    } else {
      import_name + ".thy"
    }
    local_thy_files.find(_.getFileName == Path.of(file_name).getFileName) match {
      case Some(result_path) => return Some(result_path, "LOCALLY")
      case None => None
    }
  }

  def locate_remotely(import_name: String): Option[(Path,String)] = {
    Imports.Ops.find_thy_file(import_name).retrieveNow match {
      case Some(result_path) => return Some(result_path, "REMOTE=" + import_name)
      case None => None
    }
  }

  def locate(import_name: String): (Path, String) = {
    val file_path: Option[(Path,String)] = locate_locally(import_name)
      .orElse(locate_via_thy(import_name))
      .orElse(locate_remotely(import_name))
    file_path match {
      case Some(result) => result
      case None => throw new Exception(s"Imports.locate: could not find $import_name in $work_dir")
    }
  }

  private def init_deps(): Graph[Path, Option[Theory]] = {
    var file_dep_graph: Graph[Path, Option[Theory]] = Graph.empty

    local_thy_files.foreach { thy_file_path =>
      if (debug) println(s"Imports: Adding local ${thy_file_path.toString}")
      val import_name_attempt = work_dir.getFileName.toString + "." + file_name_without_extension(thy_file_path)
      val opt_thy_to_add = if (Imports.Ops.can_get_thy(import_name_attempt).force.retrieveNow) {
        Some(Theory(import_name_attempt))
      } else None
      file_dep_graph = file_dep_graph.new_node(thy_file_path, opt_thy_to_add)
    }    

    local_thy_files.foreach { thy_file_path =>
      if (debug) println(s"Imports: Locating parents of ${thy_file_path.toString}")
      val parents = get_import_names(thy_file_path).map(locate)
      parents.foreach{ case (parent, location_method) =>
        if (debug) println(s"Imports: Adding parent ${parent.toString}")
        val init_parent_thy = if (location_method.startsWith("THY") || location_method.startsWith("REMOTE")) {
          val import_name = location_method.split("=").last
          Some(Theory(import_name))
        } else {
          file_dep_graph.get_node(thy_file_path)
        }
        file_dep_graph = file_dep_graph.default_node(parent, init_parent_thy)
        file_dep_graph = file_dep_graph.add_edge(thy_file_path, parent)
      }
    }
    
    file_dep_graph
  }

  def start(): Unit = {
    this.dep_graph = init_deps()
  }

  private def get_final_thy(thy0: Theory, thy_file_path: Path, thy_text: String, timeout_min: Int): Option[Theory] = {
    val final_thy_opt = Future {Imports.Ops.get_final_thy(thy0, thy_text).force.retrieveNow}

    try {
      val thy_result = Await.result(final_thy_opt, timeout_min.minutes)
      thy_result
    } catch {
      case _: java.util.concurrent.TimeoutException =>
        if (debug) println(s"Import ${thy_file_path.toString} timed out after $timeout_min minutes")
        None
      case e: Exception =>
        if (debug) println(s"An error occurred on import ${thy_file_path.toString}: ${e.getMessage}")
        None
    }
  }

  // assumption: all parent paths of thy_file_path already have value Some(thy)
  private def update_node(thy_file_path: Path): Unit = {
    val parent_paths = dep_graph.imm_succs(thy_file_path).toList
    val parent_thys = parent_paths.map(dep_graph.get_node) match {
      case thys if thys.forall(_.isDefined) => thys.flatten
      case _ => throw new Exception(s"Imports: None parent for theory $thy_file_path")
    }

    val thy_text = Files.readString(thy_file_path)
    val master_dir = thy_file_path.getParent()
    val header = Imports.Ops.get_header(thy_text, Position.none).retrieveNow.force
    val thy0 = Imports.Ops.begin_theory(master_dir, header, parent_thys).retrieveNow.force
    val final_thy = get_final_thy(thy0, thy_file_path, thy_text, 5)

    dep_graph = dep_graph.map_node(thy_file_path, {_ => final_thy})
  }

  // assumption: thy_paths is in topological order from oldest to youngest
  private def update_nodes(thy_paths: List[Path]): Unit = {
    thy_paths.foreach { thy_path =>
      dep_graph.get_node(thy_path) match {
        case Some(thy) => 
          // if (debug) println(s"Imports: Parent $thy_path already in import-graph")
          ()
        case None => 
          if (debug) println(s"Imports: Processing theory $thy_path")
          update_node(thy_path)
      }
    }
  }

  def get_parents(thy_file_path: Path): List[Theory] = {
    val all_ancesters = dep_graph.all_succs(List(thy_file_path)).reverse
    update_nodes(all_ancesters)
    dep_graph.imm_succs(thy_file_path).toList.map(dep_graph.get_node).flatten
  }

  def to_local_list(): List[Path] = {
    return dep_graph.all_preds(dep_graph.maximals).filter(local_thy_files.contains)
  }

  def load_all_deps(): Unit = {
    val eldests = dep_graph.maximals
    eldests.map{ elder => dep_graph.get_node(elder) match {
        case Some(thy) => ()
        case None => throw new Exception(s"Imports.load_all_deps: undefined elder $elder")
      }
    }
    val all_nodes = dep_graph.all_preds(eldests)
    update_nodes(all_nodes)
  }

  def get_start_theory(thy_file_path: Path): Theory = {
    val thy_text = Files.readString(thy_file_path)
    val master_dir = thy_file_path.getParent()
    val header = Imports.Ops.get_header(thy_text, Position.none).retrieveNow
    val thy0 = Imports.Ops.begin_theory(master_dir, header, get_parents(thy_file_path)).retrieveNow.force
    thy0
  }

  def get_end_theory(thy_file_path: Path): Theory = {
    dep_graph.get_node(thy_file_path) match {
        case Some(thy) => thy
        case None => 
          val _ = get_parents(thy_file_path)
          dep_graph.get_node(thy_file_path).get
    }
  }

  def get_opt_theory(thy_file_path: Path): Option[Theory] = dep_graph.get_node(thy_file_path)
}

object Imports extends OperationCollection {
  def apply(work_dir: String)(implicit isabelle: Isabelle): Imports = new Imports(Path.of(work_dir))(isabelle)

  val root_rgx: Regex = """session\s+"?([\w+-]+)"?\s*(\(\s*"?([\w\/-]+)"?\s*\)|in\s+"?([\w\/-]+)"?)?\s*=""".r

  // find all logics in a root file and 
  def find_logics(root_file: File): Map[String, File] = {
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

  def print_test_root_rgx(directory: Path): Unit = {
    val root_files = Files.walk(directory).toScala(Seq)
      .filter(path => Files.isRegularFile(path) && path.getFileName.toString == "ROOT")

    root_files.foreach { root_file =>
      println(s"Searching in file: $root_file")
      val content = Source.fromFile(root_file.toFile).mkString

      // Find matches using sessionRegex and print them
      val matches = root_rgx.findAllMatchIn(content).toList
      if (matches.nonEmpty) {
        matches.foreach { m =>
          val session_name = m.group(1)
          val location = Option(m.group(3)).orElse(Option(m.group(4)))
          println(s"Session: $session_name, Location: ${location.getOrElse("None")}")
        }
      } else {
        println(s"No matches found.")
      }
    }
  }

  // c.f. OperationCollection
  protected final class Ops(implicit isabelle: Isabelle) {
    val get_header = compileFunction[String, Position, TheoryHeader]("fn (text,pos) => Thy_Header.read pos text")

    val begin_theory = compileFunction[Path, TheoryHeader, List[Theory], Theory](
      "fn (path, header, parents) => Resources.begin_theory path header parents")
    
    val command_exception: MLFunction3[Boolean, Transition.T, ToplevelState, ToplevelState] =
      compileFunction[Boolean, Transition.T, ToplevelState, ToplevelState](
      "fn (int, tr, st) => Toplevel.command_exception int tr st")
    
    val init_toplevel: MLFunction0[ToplevelState] = compileFunction0[ToplevelState]("fn () => Toplevel.make_state NONE")

    val get_final_thy_alt: MLFunction2[Theory, String, Option[Theory]] =
      compileFunction[Theory, String, Option[Theory]](
      """fn (thy0, thy_file) => let
        |  val transitions =  
        |    File.read (Path.explode thy_file)
        |    |> Outer_Syntax.parse_text thy0 (K thy0) Position.start
        |    |> filter_out (fn tr => Toplevel.name_of tr = "<ignored>");
        |  val final_state = 
        |    Toplevel.make_state NONE
        |    |> fold (Toplevel.command_exception true) transitions;
        |  in Toplevel.previous_theory_of final_state end""".stripMargin)
    
    val get_final_thy: MLFunction2[Theory, String, Option[Theory]] =
      compileFunction[Theory, String, Option[Theory]](
      """fn (thy0, thy_text) => let
        |  val transitions =  
        |    Outer_Syntax.parse_text thy0 (K thy0) Position.start thy_text
        |    |> filter_out (fn tr => Toplevel.name_of tr = "<ignored>");
        |  val final_state = 
        |    Toplevel.make_state NONE
        |    |> fold (Toplevel.command_exception true) transitions;
        |  in Toplevel.previous_theory_of final_state end""".stripMargin)
    
    val toplevel_end_theory: MLFunction[ToplevelState, Theory] = compileFunction[ToplevelState, Theory]("Toplevel.end_theory Position.none")
    
    val can_get_thy: MLFunction[String, Boolean] = compileFunction[String, Boolean]("Basics.can Thy_Info.get_theory")

    val can_get_thy_file: MLFunction[String, Boolean] = compileFunction[String, Boolean]("Basics.can Resources.find_theory_file")

    val find_thy_file: MLFunction[String, Option[Path]] = compileFunction[String, Option[Path]]("Resources.find_theory_file")

    val get_base_name: MLFunction[String, String] = compileFunction("Long_Name.base_name")

    val get_theory_name: MLFunction[Theory, String] = compileFunction("Context.theory_name {long=false}")
  }

  // c.f. OperationCollection
  override protected def newOps(implicit isabelle: Isabelle) = {
    new this.Ops
  }
}