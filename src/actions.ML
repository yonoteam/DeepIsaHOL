
signature ACTIONS =
sig
  type T
  val text_of: T -> string
  val kind_of: T -> string
  val pos_of: T -> Position.T
  val apply: T -> Toplevel.state -> Toplevel.state * (exn * string) option
  val begins_thy: T -> bool
  val begins_prf: theory -> T -> bool
  val ends_prf: theory -> T -> bool
end;


structure Actions: ACTIONS =
struct

datatype T = Action of {
  text: string,
  transition: Toplevel.transition}

fun text_of (Action a) = #text a;

fun kind_of (Action a) = Toplevel.name_of (#transition a);

fun pos_of (Action a) = Toplevel.pos_of (#transition a);

fun apply (Action a) st = Toplevel.transition true (#transition a) st;

fun on_text f a = f (text_of a);

fun on_kind f a = f (kind_of a);

val begins_thy = Pred.conjunct [
  on_text (Pred.starts_with "theory"), 
  on_text (Pred.contains "begin"), 
  on_kind (curry (op =) "theory")];

fun begins_proof thy 
  = Pred.disjunct (map (on_text o Pred.starts_with) (Get.start_proof_keywords thy))

fun ends_proof thy 
  = Pred.disjunct (map (on_text o Pred.starts_with) (Get.end_proof_keywords thy))

end;