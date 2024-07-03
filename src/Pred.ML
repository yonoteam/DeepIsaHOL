
signature PRED =
sig
  val neg: ('a -> bool) -> 'a -> bool
  val conjunct: ('a -> bool) list -> ('a -> bool)
  val has_many: string * thm list -> bool
  val nth_letter_is: int -> string -> string * thm list -> bool
  val starts_with: string -> string * thm list -> bool
  val contains: string -> string * thm list -> bool
  val is_dynamic: Proof.context -> string * thm list -> bool
end;


structure Pred: PRED =
struct

fun neg P x = not (P x)

fun conjunct Ps = foldl1 (op andf) Ps;

fun has_many (_:string, thms: thm list) = length thms > 1;

fun starts_with str (name, _: thm list) = String.isPrefix str name;

fun nth_letter_is n str (name, _: thm list) = nth_string name n = str;

fun contains str (name, _: thm list) =  String.isSubstring str name;

fun is_dynamic ctxt (name:string, _: thm list) 
  = (#dynamic (Facts.retrieve (Context.Proof ctxt) (Proof_Context.facts_of ctxt) (name, Position.none)));


end;