
signature ACTIONS =
sig
  type T
  val text_of: T -> string
  val transition_of: T -> Toplevel.transition
  val kind_of: T -> string
  val pos_of: T -> Position.T
  val apply: T -> Toplevel.state -> Toplevel.state * (exn * string) option
  val begins_thy: T -> bool
  val begins_proof: theory -> T -> bool
  val ends_proof: theory -> T -> bool
  val make: theory -> string -> T list
end;


structure Actions: ACTIONS =
struct

datatype T = Action of {
  text: string,
  transition: Toplevel.transition}

fun text_of (Action a) = #text a;

fun transition_of (Action a) = #transition a;

fun kind_of (Action a) = Toplevel.name_of (#transition a);

fun pos_of (Action a) = Toplevel.pos_of (#transition a);

fun apply (Action a) st = Toplevel.transition true (#transition a) st;

fun on_text f a = f (text_of a);

fun on_kind f a = f (kind_of a);

(* Action.T -> bool *)
val begins_thy = Pred.conjunct [
  on_text (Pred.starts_with "theory"), 
  on_text (Pred.contains "begin"), 
  on_kind (curry (op =) "theory")];

(* theory -> Action.T -> bool *)
fun begins_proof thy 
  = Pred.disjunct (map (on_text o Pred.starts_with) (Get.start_proof_keywords thy))

(* theory -> Action.T -> bool *)
fun ends_proof thy 
  = Pred.disjunct (map (on_text o Pred.starts_with) (Get.end_proof_keywords thy))

(* inspired from Portal-to-Isabelle's parse_text MLFunction2 *)
(* theory -> string -> T list *)
fun make thy0 thy_text =
  let
    val trs = Get.transitions thy0 Position.start thy_text;
    fun dist tr1 tr2 = Option.valOf (Position.distance_of (Toplevel.pos_of tr1, Toplevel.pos_of tr2));
    fun make_act txt tr = Action {text=txt, transition=tr}
    fun record_acts syms [tr] = [make_act (implode syms) tr]
      | record_acts _ [] = []
      | record_acts syms (tr1 :: tr2 :: trs) = 
          let 
            val (syms1, syms2) = Library.chop (dist tr1 tr2) syms;
            val act = make_act (implode syms1) tr1;
          in act :: record_acts syms2 (tr2 :: trs) end
  in record_acts (Symbol.explode thy_text) trs end;

end;