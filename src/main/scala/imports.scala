/*
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz
 */

package isabelle_rl

import java.nio.file.{Path, Paths, Files}
import java.io.FileNotFoundException
import scala.jdk.CollectionConverters._
import de.unruh.isabelle.control.{Isabelle, OperationCollection}
import de.unruh.isabelle.mlvalue.MLValue.{compileFunction, compileFunction0}
import de.unruh.isabelle.pure.{Position, Theory, TheoryHeader, ToplevelState}
import de.unruh.isabelle.mlvalue.{AdHocConverter, MLFunction, MLFunction0, MLFunction2, MLFunction3}
import isabelle_rl.Theory_Loader.{Heap, Source, Text}
import Theory_Loader.Ops
import isabelle_rl.Graph

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._
import scala.concurrent.ExecutionContext.Implicits.global

class Imports (val work_dir: Path)(implicit isabelle: Isabelle) {
  private val debug = true

  val local_thy_files: List[Path] = {
    val files = Files.walk(work_dir).iterator().asScala
    val filtered_files = files.filter { path => Files.isRegularFile(path) && path.toString.endsWith(".thy")}
    filtered_files.toList
  }
  
  private def file_name_without_extension(file_path: Path): String = {
    val file_name = file_path.getFileName.toString
    if (file_name.contains(".")) {
      file_name.substring(0, file_name.lastIndexOf('.'))
    } else {
      file_name
    }
  }

  def locate_via_thy(import_name: String): Option[(Path,String)] = {
    if (Ops.can_get_thy(import_name).retrieveNow) {
        Some(Path.of("ISABELLE=" + import_name), "THY=" + import_name)
    } else None
  }

  def locate_locally(import_name: String): Option[(Path,String)] = {
    val file_name = if (import_name.contains(".")) {
      Ops.get_base_name(import_name).retrieveNow + ".thy"
    } else if (import_name.contains("/")) {
      Path.of(import_name + ".thy").getFileName.toString
    } else {
      import_name + ".thy"
    }
    local_thy_files.find(_.getFileName.toString == file_name) match {
      case Some(result_path) => return Some(result_path, "LOCALLY")
      case None => None
    }
  }

  def locate_remotely(import_name: String): Option[(Path,String)] = {
    Ops.find_thy_file(import_name).retrieveNow match {
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

  def init_deps(): Graph[Path, Option[Theory]] = {
    var file_dep_graph: Graph[Path, Option[Theory]] = Graph.empty

    local_thy_files.foreach { thy_file_path =>
      if (debug) println(s"Adding local ${thy_file_path.toString}")
      file_dep_graph = file_dep_graph.new_node(thy_file_path, None)
    }

    local_thy_files.foreach { thy_file_path =>
      if (debug) println(s"Processing parents of ${thy_file_path.toString}")
      val parents = Text.from_file(thy_file_path).get_imports.map(locate)
      parents.foreach{ case (parent, location_method) =>
        if (debug) println(s"Processing parent ${parent.toString}")
        val init_parent_thy = if (location_method.startsWith("THY") || location_method.startsWith("REMOTE")) {
          val import_name = location_method.split("=").last
          Some(Theory(import_name))
        } else {
            None
        }
        file_dep_graph = file_dep_graph.default_node(parent, init_parent_thy)
        file_dep_graph = file_dep_graph.add_edge(thy_file_path, parent)
      }
    }
    
    file_dep_graph
  }

  var dep_graph = init_deps()

  // assumption: all parent paths of thy_file_path already have value Some(thy)
  private def update_node(thy_file_path: Path): Unit = {
    val parent_paths = dep_graph.imm_succs(thy_file_path).toList
    val parent_thys = parent_paths.map(dep_graph.get_node) match {
      case thys if thys.forall(_.isDefined) => thys.flatten
      case _ => throw new Exception(s"Imports.update_node: None parent for theory $thy_file_path")
    }

    val thy_text = Files.readString(thy_file_path)
    val master_dir = thy_file_path.getParent()
    val header = Ops.header_read(thy_text, Position.none).retrieveNow
    val thy0 = Ops.begin_theory(master_dir, header, parent_thys).retrieveNow
    val final_thy = Ops.get_final_thy(thy0, thy_text).retrieveNow

    dep_graph = dep_graph.map_node(thy_file_path, {_ => final_thy})
  }

  // assumption: thy_paths is in topological order from oldest to youngest
  private def update_nodes(thy_paths: List[Path]): Unit = {
    thy_paths.foreach { ancester =>
      if (debug) println(s"Imports.update_nodes: Processing theory $ancester")
      dep_graph.get_node(ancester) match {
        case Some(thy) => ()
        case None => update_node(ancester)
      }
    }
  }

  def get_parents(thy_file_path: Path): List[Theory] = {
    val all_ancesters = dep_graph.all_succs(List(thy_file_path)).reverse
    update_nodes(all_ancesters)
    dep_graph.imm_succs(thy_file_path).toList.map(dep_graph.get_node).flatten
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
    val header = Ops.header_read(thy_text, Position.none).retrieveNow
    val thy0 = Ops.begin_theory(master_dir, header, get_parents(thy_file_path)).retrieveNow
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
}
