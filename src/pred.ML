
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
  val split: ('a -> bool) -> 'a list -> 'a list list
  val positions: ('a -> bool) -> 'a list -> int list
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

(* ('a -> bool) -> 'a list -> 'a list list *)
fun split _ [] = []
  | split P (x :: xs) =
  let
    fun decide y (b, []) = if P y = b then (b, [[y]]) else (not b, [[y]])
      | decide y (b, ys :: yss) =  
          if P y = b then (b, (y :: ys) :: yss) else (not b, ([y]) :: ys :: yss);
    val start = if P x then (true, [[x]]) else (false, [[x]])
  in rev (map rev (#2 (fold decide xs start))) end;

(* ('a -> bool) -> 'a list -> int list *)
fun positions P xs = 
  let
    fun count _ [] = []
      | count n (y :: ys) = if P y then n :: count (n + 1) ys else count (n + 1) ys;
  in count 0 xs end;

end;
