
signature PRED =
sig
  val neg: ('a -> bool) -> 'a -> bool
  val conjunct: ('a -> bool) list -> ('a -> bool)
  val disjunct: ('a -> bool) list -> ('a -> bool)
  val on_fst: ('a -> bool) -> 'a * 'b -> bool
  val on_snd: ('b -> bool) -> 'a * 'b -> bool
  val has_many: 'a list -> bool
  val nth_letter_is: int -> string -> string -> bool
  val starts_with: string -> string -> bool
  val contains: string -> string -> bool
  val is_dynamic: Proof.context -> string * thm list -> bool
end;


structure Pred: PRED =
struct

fun neg P x = not (P x)

fun conjunct Ps = foldl1 (op andf) Ps;

fun disjunct Ps = foldl1 (op orf) Ps;

fun on_fst P (x, _) = P x;

fun on_snd P (_, y) = P y;

fun has_many xs = length xs > 1;

fun starts_with str name = String.isPrefix str name;

fun nth_letter_is n str name = nth_string name n = str;

fun contains str name =  String.isSubstring str name;

fun is_dynamic ctxt (name:string, _: thm list) 
  = (#dynamic (Facts.retrieve (Context.Proof ctxt) (Proof_Context.facts_of ctxt) (name, Position.none)));


end;