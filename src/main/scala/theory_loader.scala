/*
  Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

This file is adapted from Dominique Unruh's scala-isabelle library's 
TheoryManager.scala at https://github.com/dominique-unruh/scala-isabelle
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
import isabelle_rl.Theory_Loader.Ops

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._
import scala.concurrent.ExecutionContext.Implicits.global

object Transition extends AdHocConverter("Toplevel.transition")

class Theory_Loader(val isa_logic: String, var path_to_isa: String, var work_dir : String) {
  // Start the scala-isabelle setup
  val setup: Isabelle.Setup = Isabelle.Setup(isabelleHome = Path.of(path_to_isa),
    sessionRoots = Nil,
    logic = isa_logic,
    workingDirectory = Path.of(work_dir)
  )
  implicit val isabelle: Isabelle = new Isabelle(setup)

  val local_thy_files: List[Path] = {
    val files = Files.walk(setup.workingDirectory).iterator().asScala
    val filteredFiles = files.filter { path => Files.isRegularFile(path) && path.toString.endsWith(".thy")}
    filteredFiles.toList
  } 
  
  def make_source(name: String): Source = {
    // look for import in the heap
    if (Ops.can_get_thy(name).retrieveNow) {
      return Theory_Loader.Heap(name)
    }
    // look for import in working directory
    val file_name = if (name.contains(".")) (Ops.get_base_name(name).retrieveNow + ".thy") else (name + ".thy")
    local_thy_files.find(_.getFileName.toString == file_name) match {
      case Some(result_path) => return Theory_Loader.Text.from_file(result_path)
      case None => ()
    }
    val result_path = setup.workingDirectory.resolve(file_name)
    if (Files.exists(result_path)) {
      return Theory_Loader.Text.from_file(result_path)
    } 
    
    // look for import in Isabelle's reach
    if (Ops.can_get_thy_file(name).retrieveNow) {
      return Theory_Loader.Heap(name)
    } else {
      throw new FileNotFoundException(s"Theory_Loader.make_source: Import $name not found.")
    }
  }

  // recursive construction of a theory 
  private def get_end_theory(source: Source)(implicit isabelle: Isabelle): Theory = source match {
    case Heap(name) => Theory(name)
    case Text(text, path, position) =>
      var toplevel = Ops.init_toplevel().force.retrieveNow
      var thy0 = begin_theory(source)
      for ((transition, text) <- Ops.parse_text(thy0, text).force.retrieveNow) {
        toplevel = Ops.command_exception(true, transition, toplevel).retrieveNow.force
      }
      Ops.toplevel_end_theory(toplevel).retrieveNow.force
  }

  def begin_theory(source: Source)(implicit isabelle: Isabelle): Theory = source match {
    case Heap(name) => Theory(name)
    case Text(text, path, position) =>
      val header = get_header(source)
      val masterDir = source.path.getParent
      Ops.begin_theory(masterDir, header, header.imports.map(make_source).map(get_end_theory)).retrieveNow
  }

  private def get_header(source: Source)(implicit isabelle: Isabelle): TheoryHeader = source match {
    case Text(text, path, position) => Ops.header_read(text, position).retrieveNow
  }
}

object Theory_Loader extends OperationCollection {
  
  // 3 kinds of Sources
  trait Source { def path : Path }

  case class Heap(name: String) extends Source {
    override def path: Path = Paths.get("INVALID")
  }
  case class File(path: Path) extends Source
  case class Text(text: String, path: Path, position: Position) extends Source {
    def get_text: String = {return text}
  }
  object Text {
    def from_file(path: Path)(implicit isabelle: Isabelle): Text = {
      val content = Files.readString(path)
      new Text(content, path, Position.none)
    }

    def apply(text: String, path: Path)(implicit isabelle: Isabelle): Text = new Text(text, path, Position.none)
  }

  // c.f. OperationCollection
  protected final class Ops(implicit isabelle: Isabelle) {
    val header_read = compileFunction[String, Position, TheoryHeader]("fn (text,pos) => Thy_Header.read pos text")

    val begin_theory = compileFunction[Path, TheoryHeader, List[Theory], Theory](
      "fn (path, header, parents) => Resources.begin_theory path header parents")
    
    val command_exception: MLFunction3[Boolean, Transition.T, ToplevelState, ToplevelState] =
      compileFunction[Boolean, Transition.T, ToplevelState, ToplevelState](
      "fn (int, tr, st) => Toplevel.command_exception int tr st")
    
    val init_toplevel: MLFunction0[ToplevelState] = compileFunction0[ToplevelState]("fn () => Toplevel.make_state NONE")

    val parse_text: MLFunction2[Theory, String, List[(Transition.T, String)]] =
      compileFunction[Theory, String, List[(Transition.T, String)]](
      """fn (thy, text) => let
        |  val transitions = Outer_Syntax.parse_text thy (K thy) Position.start text
        |  fun addtext symbols [tr] =
        |        [(tr, implode symbols)]
        |    | addtext _ [] = []
        |    | addtext symbols (tr::nextTr::trs) = let
        |        val (this,rest) = Library.chop (Position.distance_of (Toplevel.pos_of tr, Toplevel.pos_of nextTr) |> Option.valOf) symbols
        |        in (tr, implode this) :: addtext rest (nextTr::trs) end
        |  in addtext (Symbol.explode text) transitions end""".stripMargin)
    
    val toplevel_end_theory: MLFunction[ToplevelState, Theory] = compileFunction[ToplevelState, Theory]("Toplevel.end_theory Position.none")

    val can_get_thy: MLFunction[String, Boolean] = compileFunction[String, Boolean]("Basics.can Thy_Info.get_theory")

    val can_get_thy_file: MLFunction[String, Boolean] = compileFunction[String, Boolean]("Basics.can Resources.find_theory_file")

    val get_base_name: MLFunction[String, String] = compileFunction("Long_Name.base_name")
  }

  // c.f. OperationCollection
  override protected def newOps(implicit isabelle: Isabelle) = {
    new this.Ops
  }
}