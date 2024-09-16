/*
This file is adapted from Dominique Unruh's scala-isabelle library's 
TheoryManager.scala at https://github.com/dominique-unruh/scala-isabelle
 */

package isabelle_rl

import java.nio.file.{Path, Paths}
import de.unruh.isabelle.control.{Isabelle, OperationCollection}
import de.unruh.isabelle.mlvalue.MLValue.{compileFunction, compileFunction0}
import de.unruh.isabelle.pure.{Position, Theory, TheoryHeader, ToplevelState}
import de.unruh.isabelle.mlvalue.{AdHocConverter, MLFunction, MLFunction0, MLFunction2, MLFunction3}
import isabelle_rl.TheoryLoader.{Heap, Source, Text}
import TheoryLoader.Ops

// Implicits
import de.unruh.isabelle.mlvalue.Implicits._
import de.unruh.isabelle.pure.Implicits._
import scala.concurrent.ExecutionContext.Implicits.global

object Transition extends AdHocConverter("Toplevel.transition")

class TheoryLoader(val isa_logic: String, var path_to_isa: String, var work_dir : String) {
  // Start the scala-isabelle setup
  val setup: Isabelle.Setup = Isabelle.Setup(isabelleHome = Path.of(path_to_isa),
    sessionRoots = Nil,
    logic = isa_logic,
    workingDirectory = Path.of(work_dir)
  )
  implicit val isabelle: Isabelle = new Isabelle(setup)
  
  // recursive construction of a theory 
  def getTheorySource(name: String): Source = Heap(name)
  def getTheory(source: Source)(implicit isabelle: Isabelle): Theory = source match {
    case Heap(name) => Theory(name)
    case Text(text, path, position) =>
      var toplevel = Ops.init_toplevel().force.retrieveNow
      var thy0 = beginTheory(source)
      for ((transition, text) <- Ops.parse_text(thy0, text).force.retrieveNow) {
        toplevel = Ops.command_exception(true, transition, toplevel).retrieveNow.force
      }
      Ops.toplevel_end_theory(toplevel).retrieveNow.force
  }

  def beginTheory(source: Source)(implicit isabelle: Isabelle): Theory = {
    val header = getHeader(source)
    val masterDir = source.path.getParent
    Ops.begin_theory(masterDir, header, header.imports.map(getTheorySource).map(getTheory)).retrieveNow
  }
  def getHeader(source: Source)(implicit isabelle: Isabelle): TheoryHeader = source match {
    case Text(text, path, position) => Ops.header_read(text, position).retrieveNow
  }
}

object TheoryLoader extends OperationCollection {
  
  // 2 kinds of Sources
  trait Source { def path : Path }

  case class Heap(name: String) extends Source {
    override def path: Path = Paths.get("INVALID")
  }
  
  case class Text(text: String, path: Path, position: Position) extends Source
  object Text {
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

  }

  // c.f. OperationCollection
  override protected def newOps(implicit isabelle: Isabelle) = {
    new this.Ops
  }
}