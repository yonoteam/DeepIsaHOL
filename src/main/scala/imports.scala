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
import isabelle_rl.Graph

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._

class Imports (val work_dir: Path)(implicit isabelle: Isabelle) {
  private val debug = false

  override def toString(): String = {
    "Imports(read_dir=" + work_dir + ")"
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

  def locate_via_thy(import_name: String): Option[(Path,String)] = {
    if (Imports.Ops.can_get_thy(import_name).retrieveNow) {
        Some(Path.of("ISABELLE=" + import_name), "THY=" + import_name)
    } else None
  }

  def locate_locally(import_name: String): Option[(Path,String)] = {
    val file_name = if (import_name.contains(".")) {
      Imports.Ops.get_base_name(import_name).retrieveNow + ".thy"
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
      if (debug) println(s"Adding local ${thy_file_path.toString}")
      file_dep_graph = file_dep_graph.new_node(thy_file_path, None)
    }

    local_thy_files.foreach { thy_file_path =>
      if (debug) println(s"Processing parents of ${thy_file_path.toString}")
      val parents = get_import_names(thy_file_path).map(locate)
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

  private var dep_graph = init_deps()

  // assumption: all parent paths of thy_file_path already have value Some(thy)
  private def update_node(thy_file_path: Path): Unit = {
    val parent_paths = dep_graph.imm_succs(thy_file_path).toList
    val parent_thys = parent_paths.map(dep_graph.get_node) match {
      case thys if thys.forall(_.isDefined) => thys.flatten
      case _ => throw new Exception(s"Imports: None parent for theory $thy_file_path")
    }

    val thy_text = Files.readString(thy_file_path)
    val master_dir = thy_file_path.getParent()
    val header = Imports.Ops.get_header(thy_text, Position.none).retrieveNow
    val thy0 = Imports.Ops.begin_theory(master_dir, header, parent_thys).retrieveNow
    val final_thy = Imports.Ops.get_final_thy(thy0, thy_text).retrieveNow

    dep_graph = dep_graph.map_node(thy_file_path, {_ => final_thy})
  }

  // assumption: thy_paths is in topological order from oldest to youngest
  private def update_nodes(thy_paths: List[Path]): Unit = {
    thy_paths.foreach { thy_path =>
      if (debug) println(s"Imports: Processing theory $thy_path")
      dep_graph.get_node(thy_path) match {
        case Some(thy) => ()
        case None => update_node(thy_path)
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
    val header = Imports.Ops.get_header(thy_text, Position.none).retrieveNow
    val thy0 = Imports.Ops.begin_theory(master_dir, header, get_parents(thy_file_path)).retrieveNow
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

object Imports extends OperationCollection {
  def apply(work_dir: String)(implicit isabelle: Isabelle): Imports = new Imports(Path.of(work_dir))(isabelle)

  // c.f. OperationCollection
  protected final class Ops(implicit isabelle: Isabelle) {
    val get_header = compileFunction[String, Position, TheoryHeader]("fn (text,pos) => Thy_Header.read pos text")

    val begin_theory = compileFunction[Path, TheoryHeader, List[Theory], Theory](
      "fn (path, header, parents) => Resources.begin_theory path header parents")
    
    val command_exception: MLFunction3[Boolean, Transition.T, ToplevelState, ToplevelState] =
      compileFunction[Boolean, Transition.T, ToplevelState, ToplevelState](
      "fn (int, tr, st) => Toplevel.command_exception int tr st")
    
    val init_toplevel: MLFunction0[ToplevelState] = compileFunction0[ToplevelState]("fn () => Toplevel.make_state NONE")

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
  }

  // c.f. OperationCollection
  override protected def newOps(implicit isabelle: Isabelle) = {
    new this.Ops
  }
}