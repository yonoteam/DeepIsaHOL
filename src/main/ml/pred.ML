(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Useful ML operations based on predicates.
*)

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
  val cons_if: ('a -> bool) -> 'a -> 'a list -> 'a list
  val find_first: ('a -> bool) -> 'a list -> 'a option
  val split: ('a -> bool) -> 'a list -> 'a list list
  val positions: ('a -> bool) -> 'a list -> int list
  val nhds: ('a -> bool) -> int -> 'a list -> ('a * 'a list) list
  val log_fold: ('b -> bool) -> ('a -> 'b -> 'b) -> 'a list -> 'b -> 'b list * 'a list
  val take_until: ('a -> bool) -> 'a list -> 'a list
  val take_until': ('a -> bool) -> 'a list -> 'a list
  val drop_until: ('a -> bool) -> 'a list -> 'a list
  val take_while: ('a -> bool) -> 'a list -> 'a list
  val drop_while: ('a -> bool) -> 'a list -> 'a list
  val bifurcate_ifL: ('a -> bool) -> 'a list -> 'a list * 'a list
  val split_whenL: ('a -> bool) -> 'a list -> 'a list list
  val find_files: (Path.T -> bool) -> Path.T -> Path.T list
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

fun has_many xs = length xs > 1;

fun starts_with str name = String.isPrefix str name;

fun ends_with str name = String.isSuffix str name;

fun nth_letter_is n str name = nth_string name n = str;

fun contains str name =  String.isSubstring str name;

fun is_dynamic ctxt (name:string, _: thm list) 
  = (#dynamic (Facts.retrieve (Context.Proof ctxt) (Proof_Context.facts_of ctxt) (name, Position.none)));

fun cons_if P x xs = if P x then x :: xs else xs;

(** LIST OPERATORS **)

fun find_first _ [] = NONE
  | find_first P (x::xs) = if P x then SOME x else find_first P xs;

fun split _ [] = []
  | split P (x :: xs) =
  let
    fun decide y (b, []) = if P y = b then (b, [[y]]) else (not b, [[y]])
      | decide y (b, ys :: yss) =  
          if P y = b then (b, (y :: ys) :: yss) else (not b, ([y]) :: ys :: yss);
    val start = if P x then (true, [[x]]) else (false, [[x]])
  in rev (map rev (#2 (fold decide xs start))) end;

fun positions P xs = 
  let
    fun count _ [] = []
      | count n (y :: ys) = if P y then n :: count (n + 1) ys else count (n + 1) ys;
  in count 0 xs end;

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

(* while-loop *)
fun log_fold _ _ [] _ = ([], [])
  | log_fold P f (x :: xs) y = 
    let val y' = f x y in
    if P y' then
      let
        val (ys', xs') = log_fold P f xs y';
      in (y':: ys', xs') end
    else ([], x :: xs) end

fun take_until _ [] = []
  | take_until P (x :: xs) = if P x then [x] else x :: (take_until P xs);

fun take_until' _ [] = []
  | take_until' P (x :: xs) = if P x then [] else x :: (take_until' P xs);

fun drop_until _ [] = []
  | drop_until P (x :: xs) = if P x then x :: xs else drop_until P xs;

val take_while = Library.take_prefix

val drop_while = Library.drop_prefix

fun bifurcate_ifL _ [] = ([], [])
  | bifurcate_ifL P (c :: cs) = 
      if P c then ([c],cs)
      else let 
        val (xs, ys) = bifurcate_ifL P cs
      in (c::xs, ys) end;

fun split_whenL _ [] = []
  | split_whenL P (c :: cs) = 
      bifurcate_ifL P (c :: cs) 
      |> (fn (sub1, sub2) => sub1 :: (split_whenL P sub2));

fun find_files P path =
  let 
    fun sub_elems dir_path = File.read_dir dir_path
      |> map (Path.append dir_path o Path.explode);
    fun conditionally_add elem so_fars =
      if File.is_dir elem then 
        let
          val subs = sub_elems elem
        in fold conditionally_add subs so_fars end
      else if File.is_file elem andalso P elem then elem :: so_fars
      else if File.exists elem then so_fars
      else error ("Path does not exists " ^ (Path.implode elem));
  in conditionally_add path [] end;

end;
