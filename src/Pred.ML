
signature PRED =
sig
  val neg: ('a -> bool) -> 'a -> bool
  val conjunct: ('a -> bool) list -> ('a -> bool)
  val has_many: 'a * thm list -> bool
  val nth_letter_is: int -> string -> string * 'a -> bool
  val starts_with: string -> string * 'a -> bool
  val contains: string -> string * 'a -> bool
  val is_dynamic: Proof.context -> string * thm list -> bool
end;


structure Pred: PRED =
struct

fun neg P x = not (P x)

fun conjunct Ps = foldl1 (op andf) Ps;

fun has_many (_, thms: thm list) = length thms > 1;

fun starts_with str (name, _) = String.isPrefix str name;

fun nth_letter_is n str (name, _) = nth_string name n = str;

fun contains str (name, _) =  String.isSubstring str name;

fun is_dynamic ctxt (name:string, _: thm list) 
  = (#dynamic (Facts.retrieve (Context.Proof ctxt) (Proof_Context.facts_of ctxt) (name, Position.none)));


end;