
signature PRED =
sig
  val neg: ('a -> bool) -> 'a -> bool
  val conjunct: ('a -> bool) list -> ('a -> bool)
  val disjunct: ('a -> bool) list -> ('a -> bool)
  val on_fst: ('a -> bool) -> 'a * 'b -> bool
  val on_snd: ('b -> bool) -> 'a * 'b -> bool
  val is: ''a -> ''a -> bool
  val is_not: ''a -> ''a -> bool
  val has_many: 'a list -> bool
  val nth_letter_is: int -> string -> string -> bool
  val starts_with: string -> string -> bool
  val ends_with: string -> string -> bool
  val contains: string -> string -> bool
  val is_dynamic: Proof.context -> string * thm list -> bool
  val find_first: ('a -> bool) -> 'a list -> 'a option
  val split: ('a -> bool) -> 'a list -> 'a list list
  val positions: ('a -> bool) -> 'a list -> int list
  val nhds: ('a -> bool) -> int -> 'a list -> ('a * 'a list) list
  val take_until: ('a -> bool) -> 'a list -> 'a list
  val bifurcate_ifL: ('a -> bool) -> 'a list -> 'a list * 'a list
  val split_whenL: ('a -> bool) -> 'a list -> 'a list list
end;


structure Pred: PRED =
struct

(** BASIC OPERATORS **)

fun neg P x = not (P x)

fun conjunct Ps = foldl1 (op andf) Ps;

fun disjunct Ps = foldl1 (op orf) Ps;

fun on_fst P (x, _) = P x;

fun on_snd P (_, y) = P y;

fun is x y = (x = y);

fun is_not x y = not (x = y);

(** PREDICATES **)

(* has_many: 'a list -> bool *)
fun has_many xs = length xs > 1;

(* starts_with: string -> string -> bool *)
fun starts_with str name = String.isPrefix str name;

(* ends_with: string -> string -> bool *)
fun ends_with str name = String.isSuffix str name;

(* nth_letter_is: int -> string -> string -> bool *)
fun nth_letter_is n str name = nth_string name n = str;

(* contains: string -> string -> bool *)
fun contains str name =  String.isSubstring str name;

(* is_dynamic: Proof.context -> string * thm list -> bool *)
fun is_dynamic ctxt (name:string, _: thm list) 
  = (#dynamic (Facts.retrieve (Context.Proof ctxt) (Proof_Context.facts_of ctxt) (name, Position.none)));


(** LIST OPERATORS **)

(* find_first: ('a -> bool) -> 'a list -> 'a option *)
fun find_first _ [] = NONE
  | find_first P (x::xs) = if P x then SOME x else find_first P xs;

(* split: ('a -> bool) -> 'a list -> 'a list list *)
fun split _ [] = []
  | split P (x :: xs) =
  let
    fun decide y (b, []) = if P y = b then (b, [[y]]) else (not b, [[y]])
      | decide y (b, ys :: yss) =  
          if P y = b then (b, (y :: ys) :: yss) else (not b, ([y]) :: ys :: yss);
    val start = if P x then (true, [[x]]) else (false, [[x]])
  in rev (map rev (#2 (fold decide xs start))) end;

(* positions: ('a -> bool) -> 'a list -> int list *)
fun positions P xs = 
  let
    fun count _ [] = []
      | count n (y :: ys) = if P y then n :: count (n + 1) ys else count (n + 1) ys;
  in count 0 xs end;

(* nhds: ('a -> bool) -> int -> 'a list -> ('a * 'a list) list *)
(* neighbourhoods such that the center satisfies P *)
fun nhds _ _ [] = []
  | nhds _ 0 _ = []
  | nhds P r (x :: xs) =
    if r < 0 then raise Fail "negative radius"
    else let
      fun do_next y (buff, n_val_hds) =
        let
          val new_buff = if null buff then [y]
            else if length buff <= r then buff @ [y]
            else tl buff @ [y]
          fun update (n, z, nhd) = if n = r then (n, z, nhd) else (n+1, z, nhd @ [y]);
          val updated = map update n_val_hds
        in (new_buff, if P y then ((0, y, new_buff) :: updated) else updated) end;
      val (_, rev_result)  = fold do_next (x :: xs) ([], [])
    in map (fn (_, res, nhd) => (res, nhd)) (rev rev_result) end;

(* take_until: ('a -> bool) -> 'a list -> 'a list *)
fun take_until _ [] = []
  | take_until P (x :: xs) = if P x then [x] else x :: (take_until P xs)

(* bifurcate_ifL: ('a -> bool) -> 'a list -> 'a list * 'a list *)
fun bifurcate_ifL _ [] = ([], [])
  | bifurcate_ifL P (c :: cs) = 
      if P c then ([c],cs)
      else let 
        val (xs, ys) = bifurcate_ifL P cs
      in (c::xs, ys) end;

(* splitL: ('a -> bool) -> 'a list -> 'a list list *)
fun split_whenL _ [] = []
  | split_whenL P (c :: cs) = 
      bifurcate_ifL P (c :: cs) 
      |> (fn (sub1, sub2) => sub1 :: (split_whenL P sub2));

end;
